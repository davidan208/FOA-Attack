import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import wandb
wandb.setup(wandb.Settings(mode="disabled"))
from config_schema import MainConfig

# Import surrogate models and loss functions from surrogates module
from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureExtractor_ot,
)

from utils import get_run_name, setup_wandb

# Backbone mapping matching standard naming conventions
BACKBONE_MAP = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor
}

# =====================================================================
#             SECTION: LOSS FUNCTION MINIMIZATION (SOT LOSS)
# =====================================================================
class SOTLossFunction(nn.Module):
    """
    Custom Loss Function for Adversarial Feature Alignment.
    Modify this class to implement novel loss formulations (e.g., custom transport plans,
    metric learning losses, or new global/local alignment measures).
    """
    MIX_MODES = ("adaptive", "base_only", "multi_only", "uniform")
    ENC_WEIGHT_MODES = ("adaptive", "uniform")

    def __init__(self, extractors: List[nn.Module], cluster_number: int = 10,
                 mix: str = "adaptive", enc_weight: str = "adaptive"):
        super(SOTLossFunction, self).__init__()
        if mix not in self.MIX_MODES:
            raise ValueError(f"optim.mix must be one of {self.MIX_MODES}, got '{mix}'")
        if enc_weight not in self.ENC_WEIGHT_MODES:
            raise ValueError(f"optim.enc_weight must be one of {self.ENC_WEIGHT_MODES}, got '{enc_weight}'")
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number
        self.mix = mix
        self.enc_weight = enc_weight
        self.ground_truth_global = []
        self.ground_truth_local = []

        # State tracking for dynamic weighting calculation
        self.previous_loss_list = []

        # Measurement hooks (read-only; set by StepLogger). record: store detached values of the
        # last forward in self.last_record. keep_graph: also keep the live s_i tensors for
        # the gradient diagnostic in self.last_graph.
        self.record = False
        self.keep_graph = False
        self.last_record = None
        self.last_graph = None

    @torch.no_grad()
    def set_ground_truth(self, x: torch.Tensor):
        """
        Extracts and stores target features (global and local) to align with.
        """
        self.ground_truth_global.clear()
        self.ground_truth_local.clear()
        
        for model in self.extractors:
            # Extract features from current target image
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            # x_tensor: [B, dim], x_embedding: [B, num_patches, dim]
            batch_size = x_embedding.shape[0]

            # Resolve local prototypes using K-Means clustering, per sample
            centers = [
                self.get_cluster_center(x_embedding[b], x.device)
                for b in range(batch_size)
            ]
            cluster_center = torch.stack(centers, dim=0)  # [B, cluster, dim]

            self.ground_truth_global.append(x_tensor)      # [B, dim]
            self.ground_truth_local.append(cluster_center) # [B, cluster, dim]

    def get_cluster_center(self, embedding_img: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Performs K-Means clustering on the spatial patch tokens to find cluster centers.
        """
        from kmeans_pytorch import kmeans
        import sys
        from contextlib import contextmanager

        @contextmanager
        def suppress_stdout():
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                try:
                    sys.stdout = fnull
                    yield
                finally:
                    sys.stdout = old_stdout
                    
        with suppress_stdout():
            _, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device,
                iter_limit=100,
                tol=1e-4
            )
        return cluster_center.to(device)

    def Sinkhorn(self, K: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Runs the Sinkhorn algorithm to find the optimal transport coupling plan.
        """
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for _ in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        return torch.outer(r, c) * K

    def OT(self, src_dis: torch.Tensor, tgt_dis: torch.Tensor) -> torch.Tensor:
        """
        Computes Optimal Transport (OT) alignment similarity cost.
        """
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1.0 - sim
        
        # Setup uniform marginal weights
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
            
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
            
        sim_op = torch.sum(T * sim, dim=(0, 1))
        return torch.sum(sim_op)

    def forward(
        self, 
        feature_dict: Dict[int, torch.Tensor], 
        feature_local_dict: Dict[int, torch.Tensor],
        crop_features: Optional[List[Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]]] = None,
        use_mca: bool = True
    ) -> torch.Tensor:
        """
        Combines global and local feature alignment losses across the ensemble with dynamic weighting.
        Supports Multi-Crop Alignment (MCA) dynamically balanced based on optimization state.
        """
        loss_list = []
        base_terms = [[] for _ in self.extractors]   # per encoder: [(global_b, local_b)] per sample
        
        for index, model in enumerate(self.extractors):
            gt_global = self.ground_truth_global[index]   # [B, dim]
            gt_local = self.ground_truth_local[index]      # [B, cluster, dim]
            
            feat_global = feature_dict[index]              # [B, dim]
            feat_local = feature_local_dict[index]         # [B, cluster, dim]
            
            batch_size = gt_global.shape[0]
            global_alignment = 0
            local_alignment = 0
            for b in range(batch_size):
                # Compute standard OT loss (un-cropped or base image), per sample
                g_b = self.OT(gt_global[b:b + 1], feat_global[b:b + 1])
                l_b = self.OT(gt_local[b], feat_local[b])
                global_alignment = global_alignment + g_b
                local_alignment = local_alignment + l_b
                base_terms[index].append((g_b, l_b))
            global_alignment = global_alignment / batch_size
            local_alignment = local_alignment / batch_size
            
            # Combine global and local components
            model_loss = global_alignment + 0.2 * local_alignment
            loss_list.append(model_loss)
            
        # Initialize baseline state (for first optimization step)
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in loss_list]
            
        # Calculate dynamic weighting ratios
        ratios = []
        for i in range(len(self.extractors)):
            ratio = loss_list[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            ratios.append(ratio)
            
        # Calculate standard ensemble weights (focusing on struggling encoders)
        ratios_np = np.array(ratios)
        weights_softmax = np.exp(ratios_np / 1.0)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= len(self.extractors)
        
        # Save current standard losses for the next iteration's ratio computation
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = loss_list[i].detach()
            
        # Compute MCA loss if crop features are provided
        mca_loss_list = []
        crop_terms = [[] for _ in self.extractors]   # per encoder, per crop: [(global_b, local_b)]
        num_views = 0
        if use_mca and crop_features is not None and len(crop_features) > 0:
            num_views = len(crop_features)
            for index, model in enumerate(self.extractors):
                crop_losses = []
                for crop_feat_dict, crop_feat_local_dict in crop_features:
                    c_global = crop_feat_dict[index]         # [B, dim]
                    c_local = crop_feat_local_dict[index]    # [B, cluster, dim]
                    
                    gt_global = self.ground_truth_global[index]  # [B, dim]
                    gt_local = self.ground_truth_local[index]     # [B, cluster, dim]
                    
                    batch_size = gt_global.shape[0]
                    g_align = 0
                    l_align = 0
                    terms = []
                    for b in range(batch_size):
                        g_c = self.OT(gt_global[b:b + 1], c_global[b:b + 1])
                        l_c = self.OT(gt_local[b], c_local[b])
                        g_align = g_align + g_c
                        l_align = l_align + l_c
                        terms.append((g_c, l_c))
                    crop_terms[index].append(terms)
                    g_align = g_align / batch_size
                    l_align = l_align / batch_size
                    crop_losses.append(g_align + 0.2 * l_align)
                mca_loss_list.append(torch.stack(crop_losses).mean())
        else:
            mca_loss_list = loss_list  # Fallback to standard loss if MCA is not run
            
        # Dynamically balance OT vs MCA loss for each model
        combined_losses = []
        lambdas = []
        for i in range(len(self.extractors)):
            if self.mix == "adaptive":
                lambda_i = min(ratios[i], 1.0)
            elif self.mix == "base_only":
                lambda_i = 1.0
            elif self.mix == "multi_only":
                lambda_i = 0.0
            else:  # uniform: every one of the M + 1 views weighs 1 / (M + 1)
                lambda_i = 1.0 / (num_views + 1)
            lambdas.append(lambda_i)
            combined_loss = lambda_i * loss_list[i] + (1.0 - lambda_i) * mca_loss_list[i]
            combined_losses.append(combined_loss)

        if self.enc_weight == "uniform":
            weights_softmax = np.ones(len(self.extractors))
            
        # Sum dynamically weighted model losses across the ensemble
        total_loss = sum(
            weights_softmax[i] * combined_losses[i]
            for i in range(len(self.extractors))
        )

        if self.record:
            self.last_record = self._collect_record(
                loss_list, mca_loss_list, ratios, lambdas, weights_softmax,
                base_terms, crop_terms if num_views > 0 else None,
            )
        if self.keep_graph:
            self.last_graph = {
                "s": loss_list,
                "lam": lambdas,
                "w": [float(w) for w in weights_softmax],
            }
        
        return total_loss

    @staticmethod
    def _collect_record(loss_list, mca_loss_list, ratios, lambdas, weights,
                        base_terms, crop_terms) -> Dict[str, Any]:
        """Detached copies of the values this step used; one device sync for all of them."""
        n = len(loss_list)
        batch = len(base_terms[0])
        scalars = []
        for i in range(n):
            for g, l in base_terms[i]:
                scalars += [g, l]
        if crop_terms is not None:
            for i in range(n):
                for terms in crop_terms[i]:
                    for g, l in terms:
                        scalars += [g, l]
        vals = torch.stack([t.detach().float().reshape(()) for t in scalars]).cpu().tolist()

        # Per image b and encoder i: s = global + 0.2 * local on the base view (the summands of
        # loss_list[i]); views = the same score on each extra view; m = their mean.
        pos = 0
        s_img = []
        for i in range(n):
            s_img.append([vals[pos + 2 * b] + 0.2 * vals[pos + 2 * b + 1] for b in range(batch)])
            pos += 2 * batch
        views = [[[] for _ in range(batch)] for _ in range(n)]
        if crop_terms is not None:
            for i in range(n):
                for _ in crop_terms[i]:
                    for b in range(batch):
                        views[i][b].append(vals[pos + 2 * b] + 0.2 * vals[pos + 2 * b + 1])
                    pos += 2 * batch
        m_img = [[sum(v) / len(v) if v else s_img[i][b] for b, v in enumerate(views[i])] for i in range(n)]
        return {
            "r": [float(x) for x in ratios], "lam": [float(x) for x in lambdas],
            "w": [float(x) for x in weights], "s": s_img, "m": m_img, "views": views, "batch": batch,
        }


class StepLogger:
    """
    Read-only measurement for TASK_lambda_logging.md. It only reads values the attack already
    computed: no extra random draws, no extra forward pass.

    <log_dir>/lam_log_{img:03d}.jsonl, one file per image:
      {"img", "t", "enc", "s", "m", "r", "lam", "w", "views"}   every step, enc 0/1/2 = backbone order
      {"img", "t", "type": "grad", "flip", "share", "zero_base", "flip_nz"}   every diag_every steps
    s, m, views are this image's scores. r, lam, w are exactly what the update used: the code
    computes them from the batch-mean score, so with batch_size > 1 they are shared by the batch.
    zero_base = fraction of pixels whose base-only gradient is exactly 0 (outside the base crop);
    flip_nz = flip restricted to the other pixels.
    """

    def __init__(self, ensemble_loss: "SOTLossFunction", log_dir: str,
                 diag_grad: bool = False, diag_every: int = 10):
        ensure_dir(log_dir)
        self.loss = ensemble_loss
        self.log_dir = log_dir
        self.diag_grad = diag_grad
        self.diag_every = max(1, int(diag_every))
        self.indices: List[int] = []
        self.files = []
        ensemble_loss.record = True
        print(f"  [StepLogger] writing to {log_dir} (diag_grad={diag_grad}, every {self.diag_every} steps)")

    def start_batch(self, indices: List[int]):
        self._close_files()
        self.indices = list(indices)
        self.files = [open(os.path.join(self.log_dir, f"lam_log_{img:03d}.jsonl"), "a", encoding="utf-8")
                      for img in self.indices]

    def _diag_now(self, t: int) -> bool:
        return self.diag_grad and t % self.diag_every == 0

    def before_forward(self, t: int):
        self.loss.keep_graph = self._diag_now(t)

    def grad(self, loss: torch.Tensor, delta: torch.Tensor, t: int) -> torch.Tensor:
        """Same gradient the attack uses; on diagnosed steps also the base-only gradient."""
        diag = self._diag_now(t)
        g_full = torch.autograd.grad(loss, delta, create_graph=False, retain_graph=diag)[0]
        diag_rows = None
        if diag:
            graph = self.loss.last_graph
            j_base = sum(w * lam * s for s, lam, w in zip(graph["s"], graph["lam"], graph["w"]))
            g_base = torch.autograd.grad(j_base, delta)[0]
            with torch.no_grad():
                flat_full = g_full.flatten(1)  # per image
                flat_base = g_base.flatten(1)
                differs = (torch.sign(flat_full) != torch.sign(flat_base)).float()
                nonzero = (flat_base != 0).float()
                flip = differs.mean(dim=1)
                share = (flat_full - flat_base).norm(dim=1) / (flat_full.norm(dim=1) + 1e-12)
                zero_base = 1.0 - nonzero.mean(dim=1)
                flip_nz = (differs * nonzero).sum(dim=1) / nonzero.sum(dim=1).clamp_min(1.0)
                diag_rows = torch.stack([flip, share, zero_base, flip_nz], dim=1).cpu().tolist()
            del g_base, j_base
        self.loss.last_graph = None
        self.loss.keep_graph = False
        self.write(t, diag_rows)
        return g_full

    def write(self, t: int, diag_rows=None):
        rec = self.loss.last_record
        if rec is None:
            return
        if rec["batch"] != len(self.indices):
            raise RuntimeError(f"StepLogger: batch of {rec['batch']} but {len(self.indices)} indices")
        for b, (img, f) in enumerate(zip(self.indices, self.files)):
            lines = [json.dumps({
                "img": img, "t": t, "enc": i,
                "s": rec["s"][i][b], "m": rec["m"][i][b], "r": rec["r"][i],
                "lam": rec["lam"][i], "w": rec["w"][i], "views": rec["views"][i][b],
            }) for i in range(len(rec["r"]))]
            if diag_rows is not None:
                fl, sh, zb, fnz = diag_rows[b]
                lines.append(json.dumps({"img": img, "t": t, "type": "grad", "flip": fl, "share": sh,
                                         "zero_base": zb, "flip_nz": fnz}))
            f.write("\n".join(lines) + "\n")
            f.flush()
        self.loss.last_record = None

    def _close_files(self):
        for f in self.files:
            f.close()
        self.files = []

    def close(self):
        self._close_files()

# =====================================================================


def _numeric_sort_key(filename: str) -> int:
    """
    Extract numeric index from filename for proper numeric sorting.
    E.g., '123.png' -> 123, 'img_5.jpg' -> 5
    Falls back to 0 if no number is found.
    """
    import re
    name_noext = os.path.splitext(filename)[0]
    numbers = re.findall(r'\d+', name_noext)
    if numbers:
        return int(numbers[-1])
    return 0


class RobustImageDataset(torch.utils.data.Dataset):
    """
    A robust image dataset that handles both:
      1. Flat directory layout: images directly in the root folder
      2. Class-based subdirectory layout: images nested inside subfolders
    
    Images are ALWAYS sorted numerically by index (0, 1, 2, 3, ... not 0, 1, 10, 100, ...).
    """
    def __init__(self, root_dir: str, transform: Any = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
            
        subdirs = [
            os.path.join(root_dir, d) 
            for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
        ]
        
        if len(subdirs) > 0:
            subdirs_sorted = sorted(subdirs, key=lambda x: _numeric_sort_key(os.path.basename(x)))
            print(f"  [RobustDataset] Found {len(subdirs_sorted)} subdirectories in '{root_dir}'. Loading nested images (numeric order).")
            for class_idx, subdir in enumerate(subdirs_sorted):
                all_files = []
                for root, _, files in os.walk(subdir):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in valid_extensions:
                            all_files.append(os.path.join(root, file))
                all_files.sort(key=lambda x: _numeric_sort_key(os.path.basename(x)))
                for filepath in all_files:
                    self.samples.append((filepath, class_idx))
        else:
            print(f"  [RobustDataset] No subdirectories found in '{root_dir}'. Loading images directly (numeric order).")
            all_files = []
            for file in os.listdir(root_dir):
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    all_files.append(file)
            all_files.sort(key=_numeric_sort_key)
            for file in all_files:
                self.samples.append((os.path.join(root_dir, file), 0))
                    
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image files found in path: {root_dir}")
        
        print(f"  [RobustDataset] Loaded {len(self.samples)} images. First: {os.path.basename(self.samples[0][0])}, Last: {os.path.basename(self.samples[-1][0])}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {path}: {e}")
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, path


def to_tensor(pic: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch Tensor with standard float range [0.0, 1.0]."""
    import numpy as np
    img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


def ensure_dir(path: str):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def get_surrogate_models(cfg: MainConfig, cluster_number: int) -> Tuple[torch.nn.Module, List[torch.nn.Module], torch.nn.Module]:
    """
    Load specified surrogate models dynamically and initialize ensemble extractor and custom SOT loss.
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified.")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name}. Options are: {list(BACKBONE_MAP.keys())}")
        
        model_class = BACKBONE_MAP[backbone_name]
        print(f"  [Loader] Loading model: {backbone_name} on device: {cfg.model.device}...")
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor_ot(models, cluster_number=cluster_number)
    else:
        ensemble_extractor = models[0]

    ensemble_loss = SOTLossFunction(
        models, cluster_number=cluster_number,
        mix=cfg.optim.mix, enc_weight=cfg.optim.enc_weight,
    )
    
    return ensemble_extractor, models, ensemble_loss


def log_metrics(pbar, metrics: Dict[str, float], img_index: int, epoch: Optional[int] = None):
    """Log metrics to the progress bar."""
    pbar_metrics = {
        k: f"{v:.5f}" if "loss" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)


def get_similarity_loss(
    cfg: MainConfig,
    ensemble_extractor: torch.nn.Module,
    ensemble_loss: torch.nn.Module, 
    image: torch.Tensor, 
    source_crop: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Computes similarity loss dynamically, supporting both standard OT loss and Multi-Crop Alignment (MCA).
    """
    if source_crop is not None and cfg.model.use_source_crop:
        base_image = source_crop(image)
    else:
        base_image = image
        
    outputs = ensemble_extractor(base_image)
    
    crop_features = []
    if cfg.optim.use_mca and cfg.optim.num_crops > 0 and source_crop is not None:
        for _ in range(cfg.optim.num_crops):
            cropped_image = source_crop(image)
            c_outputs = ensemble_extractor(cropped_image)
            if isinstance(c_outputs, tuple) and len(c_outputs) == 2:
                crop_features.append(c_outputs)
                
    if isinstance(outputs, tuple) and len(outputs) == 2:
        features, features_local = outputs
        return ensemble_loss(
            features, 
            features_local, 
            crop_features=crop_features, 
            use_mca=cfg.optim.use_mca
        )
    else:
        return ensemble_loss(outputs)


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: torch.nn.Module,
    ensemble_loss: torch.nn.Module,
    source_crop: Optional[torch.nn.Module],
    target_crop: Optional[torch.nn.Module],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
) -> torch.Tensor:
    """Perform FGSM attack to generate adversarial perturbations."""
    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc=f"FGSM [idx={img_index}]")
    logger = getattr(ensemble_loss, "step_logger", None)

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        if logger is not None:
            logger.before_forward(epoch)
        adv_image = image_org + delta
        loss = get_similarity_loss(cfg, ensemble_extractor, ensemble_loss, adv_image, source_crop if cfg.model.use_source_crop else None)

        metrics = {
            "loss": loss.item(),
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, metrics, img_index, epoch)

        if logger is not None:
            grad = logger.grad(loss, delta, epoch)
        else:
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    return torch.clamp(adv_image / 255.0, 0.0, 1.0)


def mifgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: torch.nn.Module,
    ensemble_loss: torch.nn.Module,
    source_crop: Optional[torch.nn.Module],
    target_crop: Optional[torch.nn.Module],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
) -> torch.Tensor:
    """Perform MI-FGSM attack with momentum."""
    delta = torch.zeros_like(image_org, requires_grad=True)
    momentum = torch.zeros_like(image_org, requires_grad=False)
    pbar = tqdm(range(cfg.optim.steps), desc=f"MI-FGSM [idx={img_index}]")
    logger = getattr(ensemble_loss, "step_logger", None)

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        if logger is not None:
            logger.before_forward(epoch)
        adv_image = image_org + delta
        loss = get_similarity_loss(cfg, ensemble_extractor, ensemble_loss, adv_image, source_crop if cfg.model.use_source_crop else None)

        metrics = {
            "loss": loss.item(),
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, metrics, img_index, epoch)

        if logger is not None:
            grad = logger.grad(loss, delta, epoch)
        else:
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(momentum),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    return torch.clamp(adv_image / 255.0, 0.0, 1.0)


def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: torch.nn.Module,
    ensemble_loss: torch.nn.Module,
    source_crop: Optional[torch.nn.Module],
    target_crop: Optional[torch.nn.Module],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
) -> torch.Tensor:
    """Perform PGD attack using Adam optimizer."""
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)
    pbar = tqdm(range(cfg.optim.steps), desc=f"PGD [idx={img_index}]")
    logger = getattr(ensemble_loss, "step_logger", None)

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta
        similarity = get_similarity_loss(cfg, ensemble_extractor, ensemble_loss, adv_image, source_crop if cfg.model.use_source_crop else None)
        
        loss = -similarity

        metrics = {
            "loss": similarity.item(),
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, metrics, img_index, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if logger is not None:
            logger.write(epoch)

        delta.data = torch.clamp(
            delta,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    return torch.clamp(adv_image / 255.0, 0.0, 1.0)


def resolve_device(device_str: str) -> str:
    """
    Checks if the requested device is available. Falls back to CPU if CUDA is not available.
    """
    if "cuda" in device_str:
        if torch.cuda.is_available():
            try:
                parts = device_str.split(":")
                if len(parts) > 1:
                    idx = int(parts[1])
                    if idx < torch.cuda.device_count():
                        return device_str
                    else:
                        print(f"  [Device] Requested '{device_str}' out of range ({torch.cuda.device_count()} GPUs). Falling back to 'cuda:0'.")
                        return "cuda:0"
                return device_str
            except ValueError:
                return "cuda:0"
        else:
            print("  [Device] CUDA not available. Falling back to 'cpu'.")
            return "cpu"
    return "cpu"


def get_completed_indices(run_dir: str) -> set:
    """
    Scan this run's own output folder for .png results already generated.
    Only this folder is scanned: each (attack, seed) variant has its own folder and
    must never resume from another variant's results.
    Returns a set of integer indices that have been completed.
    """
    completed = set()
    import re

    if not os.path.exists(run_dir):
        return completed

    for root, _, files in os.walk(run_dir):
        for file in files:
            if file.lower().endswith('.png'):
                name_noext = os.path.splitext(file)[0]
                numbers = re.findall(r'\d+', name_noext)
                if numbers:
                    completed.add(int(numbers[-1]))

    if len(completed) > 0:
        print(f"  [Resume] Found existing results in: {run_dir}")

    return completed


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    # Experiment seed for reproducible random crops and K-means initialization.
    # seed=null leaves PyTorch unseeded, reproducing the original default behaviour.
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)  # K-means picks its initial centres with np.random
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("SOTAttack: Starting Adversarial Example Generation Stage")
    print(f"Experiment seed: {seed if seed is not None else 'default (unseeded)'}")
    print("=" * 60)
    
    # 0. Resolve device
    device = resolve_device(cfg.model.device)
    OmegaConf.update(cfg, "model.device", device)
    print(f"Active computation device: {device}")
    
    # 1. Resolve absolute paths
    cle_data_path = hydra.utils.to_absolute_path(cfg.data.cle_data_path)
    tgt_data_path = hydra.utils.to_absolute_path(cfg.data.tgt_data_path)
    output_dir = hydra.utils.to_absolute_path(cfg.data.output)
    
    # 2. Setup transforms
    transform_fn = transforms.Compose([
        transforms.Resize(cfg.model.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: to_tensor(img)),
    ])
    
    # 3. Load datasets (numeric order: 0, 1, 2, 3, ...)
    print("\nInitializing datasets...")
    clean_dataset = RobustImageDataset(cle_data_path, transform=transform_fn)
    target_dataset = RobustImageDataset(tgt_data_path, transform=transform_fn)
    
    # 4. Load surrogate models
    cluster_num = 10
    print(f"\nInitializing surrogate feature extractors (cluster_number={cluster_num})...")
    try:
        ensemble_extractor, models, ensemble_loss = get_surrogate_models(cfg, cluster_num)
        print("Surrogate models loaded successfully.")
    except Exception as e:
        print(f"Error loading surrogate models: {e}")
        return
        
    # 5. Resolve source and target crops
    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop
        else torch.nn.Identity()
    )

    # Name the output subdirectory after this run's attack method, seed and config hash
    run_name = get_run_name(cfg)
    
    # 6. Resolve attack function
    attack_fn_map = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
    }
    attack_type = cfg.attack
    attack_fn = attack_fn_map.get(attack_type, pgd_attack)
    print(f"\nUsing attack method: {attack_type.upper()}")

    logger = None
    if cfg.log.lambda_log or cfg.log.diag_grad:
        if cfg.log.diag_grad and attack_type not in ("fgsm", "mifgsm"):
            raise ValueError("log.diag_grad supports attack=fgsm or mifgsm only")
        logger = StepLogger(
            ensemble_loss, os.path.join(output_dir, "logs"),
            diag_grad=cfg.log.diag_grad, diag_every=cfg.log.diag_every,
        )
        ensemble_loss.step_logger = logger
    print(f"Loss mixing: mix={cfg.optim.mix}, enc_weight={cfg.optim.enc_weight}")

    # 7. RESUME: Scan this run's output directory for already-completed images
    save_base_dir = os.path.join(output_dir, "img", run_name)
    print(f"Run folder: {os.path.join(cfg.data.output, 'img', run_name)}")
    completed_indices = get_completed_indices(save_base_dir)
    
    if len(completed_indices) > 0:
        print(f"\n{'='*60}")
        print(f"  RESUME MODE: Found {len(completed_indices)} already-generated images.")
        print(f"  Completed indices: {sorted(completed_indices)[:10]}{'...' if len(completed_indices) > 10 else ''}")
        print(f"  Will skip these and continue from where we left off.")
        print(f"{'='*60}")
    else:
        print(f"\n  No previous results found in '{save_base_dir}'. Starting from index 0.")

    # 8. Sequential processing loop: index 0, 1, 2, 3, ...
    num_samples = min(cfg.data.num_samples, len(clean_dataset), len(target_dataset))
    count_generated = 0
    count_skipped = 0
    
    print(f"\n  Total images to process: {num_samples}")
    print(f"  Processing order: sequential by index (0, 1, 2, 3, ...)")
    print(f"  Batch size: {cfg.data.batch_size}")
    print("")
    
    batch_indices = []
    
    for img_index in range(num_samples):
        # RESUME: skip already completed
        if img_index in completed_indices:
            count_skipped += 1
            continue
        
        batch_indices.append(img_index)
        
        # Process when batch is full OR last image
        if len(batch_indices) == cfg.data.batch_size or img_index == num_samples - 1:
            if len(batch_indices) == 0:
                continue
                
            # Load batch by indices
            batch_org = []
            batch_tgt = []
            batch_paths_org = []
            
            for idx in batch_indices:
                img_org, _, path_org = clean_dataset[idx]
                img_tgt, _, path_tgt = target_dataset[idx]
                batch_org.append(img_org)
                batch_tgt.append(img_tgt)
                batch_paths_org.append(path_org)
            
            image_org = torch.stack(batch_org).to(cfg.model.device)
            image_tgt = torch.stack(batch_tgt).to(cfg.model.device)
            
            idx_str = f"{batch_indices[0]}-{batch_indices[-1]}" if len(batch_indices) > 1 else str(batch_indices[0])
            print(f"\n[Index {idx_str}] Generating Adversarial Sample(s) | Progress: {count_generated + count_skipped + len(batch_indices)}/{num_samples}")
            
            if logger is not None:
                logger.start_batch(batch_indices)

            # Run attack
            adv_images = attack_fn(
                cfg=cfg,
                ensemble_extractor=ensemble_extractor,
                ensemble_loss=ensemble_loss,
                source_crop=source_crop,
                target_crop=target_crop,
                img_index=batch_indices[0],
                image_org=image_org,
                image_tgt=image_tgt,
            )

            # Save adversarial images
            for batch_pos, idx in enumerate(batch_indices):
                src_path = batch_paths_org[batch_pos]
                folder = os.path.basename(os.path.dirname(src_path))
                name = os.path.basename(src_path)
                
                folder_to_save = os.path.join(save_base_dir, folder)
                ensure_dir(folder_to_save)
                
                name_noext = os.path.splitext(name)[0]
                save_path = os.path.join(folder_to_save, name_noext + ".png")
                torchvision.utils.save_image(adv_images[batch_pos], save_path)
                print(f"  Saved: index={idx} -> {folder}/{name_noext}.png")
                count_generated += 1
            
            batch_indices = []
    
    if logger is not None:
        logger.close()

    print(f"\n{'='*60}")
    print(f"  Adversarial example generation complete!")
    print(f"  Total generated this run: {count_generated}")
    print(f"  Skipped (already existed): {count_skipped}")
    print(f"  Total in output directory: {count_generated + len(completed_indices)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
