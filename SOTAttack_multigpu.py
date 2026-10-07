"""SOTAttack on several GPUs: the same attack, with the source views spread over devices.

SOTAttack.py keeps every surrogate encoder and every source view on one device, so a
large M with batch 16 can exceed the memory of one GPU. This entry point runs
SOTAttack.py unchanged except for where the source views are encoded:

  * the primary device (model.device) holds the surrogates, the target features,
    the perturbation and the loss, exactly as in SOTAttack.py;
  * every other visible GPU holds a replica of the surrogate ensemble;
  * view j (j = 0 is the base view, j = 1..M the extra crops) is encoded on one of the
    devices, including its K-means prototypes; the features are then copied back to
    the primary device, so autograd sends the gradient back across.

By default the views are split in proportion to the free memory of each GPU, measured
once every replica is loaded (largest-remainder rounding, so the counts add up to M + 1).
Contiguous views go to the same device, base first: with M = 3 and 73 GiB vs 38 GiB
free, base, crop1, crop2 go to the first GPU and crop3 to the second. SOT_VIEW_SPLIT
overrides this: "n0,n1,..." gives the number of views per GPU in CUDA_VISIBLE_DEVICES
order, "roundrobin" sends view j to GPU j mod n. After the first optimisation step the
peak memory of every GPU is printed, so the split can be checked.

Crops, their sampling order, K-means, transport plans, adaptive coefficients and the
update are those of SOTAttack.py, so the attack itself does not change; only the
activation memory is split. Views are still encoded one after another, so it is not
faster than one GPU. Output folders, run names and resume are as in SOTAttack.py.

Pick the GPUs with CUDA_VISIBLE_DEVICES and pass the usual Hydra overrides, e.g.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
    python SOTAttack_multigpu.py --config-name=ensemble_3models \
        attack=fgsm seed=null optim.use_mca=true optim.num_crops=3 data.batch_size=16 ...
"""

import os

import hydra
import torch

import SOTAttack as sot

# One ensemble extractor per device; index 0 is the one SOTAttack.py builds on model.device.
_view_extractors = []
# _view_map[j] = index into _view_extractors of the device that encodes view j.
_view_map = []
_calls = 0


def counts_by_free_memory(free, num_views):
    """Views per device in proportion to free memory, rounded by largest remainder."""
    total = sum(free)
    if total <= 0:
        return [num_views] + [0] * (len(free) - 1)
    shares = [num_views * f / total for f in free]
    counts = [int(s) for s in shares]
    by_remainder = sorted(range(len(free)), key=lambda d: (shares[d] - counts[d], free[d]), reverse=True)
    for d in by_remainder[:num_views - sum(counts)]:
        counts[d] += 1
    return counts


def build_view_map(num_devices, num_views, spec=None, free=None):
    """Device index of every view.

    spec None/"auto": contiguous blocks sized by free memory (round-robin if free is None);
    "roundrobin": view j on device j mod n; "n0,n1,...": contiguous blocks of these sizes.
    """
    spec = (spec or "auto").strip().lower()
    if spec == "roundrobin" or (spec == "auto" and free is None):
        return [j % num_devices for j in range(num_views)]
    if spec == "auto":
        counts = counts_by_free_memory(free, num_views)
    else:
        counts = [int(c) for c in spec.split(",")]
        if len(counts) != num_devices or any(c < 0 for c in counts) or sum(counts) != num_views:
            raise ValueError(
                f"SOT_VIEW_SPLIT={spec!r} must give {num_devices} non-negative counts summing to "
                f"{num_views} views (base + {num_views - 1} crops)."
            )
    return [d for d, c in enumerate(counts) for _ in range(c)]


_single_gpu_get_surrogate_models = sot.get_surrogate_models


def get_surrogate_models(cfg, cluster_number):
    """SOTAttack.get_surrogate_models, plus a replica of the ensemble on every other GPU."""
    ensemble_extractor, models, ensemble_loss = _single_gpu_get_surrogate_models(cfg, cluster_number)
    _view_extractors[:] = [ensemble_extractor]

    primary = torch.device(cfg.model.device)
    if primary.type == "cuda" and primary.index is None:
        primary = torch.device("cuda", torch.cuda.current_device())

    if primary.type == "cuda" and cfg.model.ensemble:
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            if device == primary:
                continue
            print(f"  [MultiGPU] Loading a replica of the ensemble on {device}...")
            replicas = [
                sot.BACKBONE_MAP[name]().eval().to(device).requires_grad_(False)
                for name in cfg.model.backbone
            ]
            _view_extractors.append(sot.EnsembleFeatureExtractor_ot(replicas, cluster_number=cluster_number))

    devices = [str(_device_of(e)) for e in _view_extractors]
    if len(devices) == 1:
        print(f"  [MultiGPU] Only one GPU visible ({devices[0]}): running as SOTAttack.py.")
    num_views = 1 + (cfg.optim.num_crops if cfg.optim.use_mca else 0)
    spec = os.environ.get("SOT_VIEW_SPLIT", "auto") if len(devices) > 1 else "roundrobin"
    free = None
    if len(devices) > 1:
        # Measured after every replica is loaded, so the weights are already accounted for.
        free = [torch.cuda.mem_get_info(_device_of(e))[0] for e in _view_extractors]
        print("  [MultiGPU] free memory: " + ", ".join(
            f"{dev} {f / 2**30:.1f} GiB" for dev, f in zip(devices, free)))
    _view_map[:] = build_view_map(len(devices), num_views, spec, free)
    counts = [_view_map.count(d) for d in range(len(devices))]
    mapping = ", ".join(
        f"{'base' if j == 0 else f'crop{j}'}->{devices[d]}" for j, d in enumerate(_view_map)
    )
    print(f"  [MultiGPU] {len(devices)} device(s): {', '.join(devices)} | split ({spec}): "
          f"{','.join(map(str, counts))} | views: {mapping}")
    return ensemble_extractor, models, ensemble_loss


def _device_of(module):
    return next(module.parameters()).device


def _encode(view, j):
    """Encode view j on its assigned device and return its features on the view's own device."""
    d = _view_map[j] if j < len(_view_map) else j % len(_view_extractors)
    extractor = _view_extractors[d]
    device = _device_of(extractor)
    outputs = extractor(view.to(device))
    if device == view.device:
        return outputs
    features, features_local = outputs
    return (
        {i: t.to(view.device) for i, t in features.items()},
        {i: t.to(view.device) for i, t in features_local.items()},
    )


def _report_peak_memory():
    """Peak memory per GPU so far; called at step 2, so it covers a full forward + backward."""
    peaks = []
    for e in _view_extractors:
        device = _device_of(e)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            free, total = torch.cuda.mem_get_info(device)
            peaks.append(f"{device}: peak {peak:.1f} GiB (free now {free / 2**30:.1f} of {total / 2**30:.1f} GiB)")
    if peaks:
        print(f"\n  [MultiGPU] after step 1: {' | '.join(peaks)}")


def get_similarity_loss(cfg, ensemble_extractor, ensemble_loss, image, source_crop=None):
    """SOTAttack.get_similarity_loss with every view encoded on its assigned device."""
    global _calls
    if not _view_extractors or _view_extractors[0] is not ensemble_extractor:
        _view_extractors[:] = [ensemble_extractor]
        _view_map[:] = []
    _calls += 1
    if _calls == 2:
        _report_peak_memory()

    if source_crop is not None and cfg.model.use_source_crop:
        base_image = source_crop(image)
    else:
        base_image = image

    outputs = _encode(base_image, 0)

    crop_features = []
    if cfg.optim.use_mca and cfg.optim.num_crops > 0 and source_crop is not None:
        for j in range(1, cfg.optim.num_crops + 1):
            cropped_image = source_crop(image)
            c_outputs = _encode(cropped_image, j)
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


# SOTAttack's attack loop and main() look these up as module globals at call time.
sot.get_surrogate_models = get_surrogate_models
sot.get_similarity_loss = get_similarity_loss


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg):
    # Run SOTAttack's own main() body on the config composed here.
    getattr(sot.main, "__wrapped__", sot.main)(cfg)


if __name__ == "__main__":
    main()
