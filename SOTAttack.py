import os
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

from utils import hash_training_config, setup_wandb

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
    def __init__(self, extractors: List[nn.Module], cluster_number: int = 10):
        super(SOTLossFunction, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number
        self.ground_truth_global = []
        self.ground_truth_local = []
        
        # State tracking for dynamic weighting calculation
        self.previous_loss_list = []

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
                global_alignment = global_alignment + self.OT(gt_global[b:b + 1], feat_global[b:b + 1])
                local_alignment = local_alignment + self.OT(gt_local[b], feat_local[b])
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
        import numpy as np
        ratios_np = np.array(ratios)
        weights_softmax = np.exp(ratios_np / 1.0)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= len(self.extractors)
        
        # Save current standard losses for the next iteration's ratio computation
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = loss_list[i].detach()
            
        # Compute MCA loss if crop features are provided
        mca_loss_list = []
        if use_mca and crop_features is not None and len(crop_features) > 0:
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
                    for b in range(batch_size):
                        g_align = g_align + self.OT(gt_global[b:b + 1], c_global[b:b + 1])
                        l_align = l_align + self.OT(gt_local[b], c_local[b])
                    g_align = g_align / batch_size
                    l_align = l_align / batch_size
                    crop_losses.append(g_align + 0.2 * l_align)
                mca_loss_list.append(torch.stack(crop_losses).mean())
        else:
            mca_loss_list = loss_list  # Fallback to standard loss if MCA is not run
            
        # Dynamically balance OT vs MCA loss for each model
        combined_losses = []
        for i in range(len(self.extractors)):
            lambda_i = min(ratios[i], 1.0)
            combined_loss = lambda_i * loss_list[i] + (1.0 - lambda_i) * mca_loss_list[i]
            combined_losses.append(combined_loss)
            
        # Sum dynamically weighted model losses across the ensemble
        total_loss = sum(
            weights_softmax[i] * combined_losses[i]
            for i in range(len(self.extractors))
        )
        
        return total_loss

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

    ensemble_loss = SOTLossFunction(models, cluster_number=cluster_number)
    
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

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta
        loss = get_similarity_loss(cfg, ensemble_extractor, ensemble_loss, adv_image, source_crop if cfg.model.use_source_crop else None)

        metrics = {
            "loss": loss.item(),
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, metrics, img_index, epoch)

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

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta
        loss = get_similarity_loss(cfg, ensemble_extractor, ensemble_loss, adv_image, source_crop if cfg.model.use_source_crop else None)

        metrics = {
            "loss": loss.item(),
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, metrics, img_index, epoch)

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


def get_completed_indices(output_dir: str) -> set:
    """
    Scan output directory to find already-generated adversarial images.
    Returns a set of integer indices that have been completed.
    """
    completed = set()
    import re
    
    if not os.path.exists(output_dir):
        return completed
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith('.png'):
                name_noext = os.path.splitext(file)[0]
                numbers = re.findall(r'\d+', name_noext)
                if numbers:
                    completed.add(int(numbers[-1]))
    
    return completed


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models_100")
def main(cfg: MainConfig):
    print("=" * 60)
    print("SOTAttack: Starting Adversarial Example Generation Stage")
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

    # Get configuration hash for output subdirectory
    config_hash = hash_training_config(cfg)
    
    # 6. Resolve attack function
    attack_fn_map = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
    }
    attack_type = cfg.attack
    attack_fn = attack_fn_map.get(attack_type, pgd_attack)
    print(f"\nUsing attack method: {attack_type.upper()}")

    # 7. RESUME: Scan output directory for already-completed images
    save_base_dir = os.path.join(output_dir, "img", config_hash)
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
    
    print(f"\n{'='*60}")
    print(f"  Adversarial example generation complete!")
    print(f"  Total generated this run: {count_generated}")
    print(f"  Skipped (already existed): {count_skipped}")
    print(f"  Total in output directory: {count_generated + len(completed_indices)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
