"""Shared utilities for adversarial attack and text generation models."""

import os
import json
import yaml
import hashlib
import base64
from typing import Dict, Any, List, Optional, Union
from omegaconf import OmegaConf
import wandb
from config_schema import MainConfig


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        str: MD5 hash of the config parameters
    """
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)
        
    # Create config dict with converted values
    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "use_target_crop": bool(cfg.model.use_target_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "attack": cfg.attack,
    }
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        cfg: Configuration object containing wandb settings
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=tags,
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_seed_label(cfg: MainConfig) -> str:
    """Readable label for the run seed ('default' when PyTorch is left unseeded)."""
    seed = getattr(cfg, "seed", None)
    return "default" if seed is None else str(int(seed))


def get_run_name(cfg: MainConfig) -> str:
    """Folder name identifying one run: attack method, seed, and config hash.

    Two runs that differ only by seed produce different names, so each variant
    keeps its own results folder instead of overwriting or resuming the other.
    """
    return f"{cfg.attack}_seed{get_seed_label(cfg)}_{hash_training_config(cfg)[:8]}"


def get_output_paths(cfg: MainConfig, config_hash: Optional[str] = None) -> Dict[str, str]:
    """Get dictionary of output paths based on config.

    Args:
        cfg: Configuration object
        config_hash: Legacy hash-only folder name, used as a fallback for runs
            generated before folders were named per attack/seed

    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    run_name = get_run_name(cfg)
    legacy_name = config_hash if config_hash is not None else hash_training_config(cfg)

    img_new = os.path.join(cfg.data.output, "img", run_name)
    desc_new = os.path.join(cfg.data.output, "description", run_name)
    img_legacy = os.path.join(cfg.data.output, "img", legacy_name)
    desc_legacy = os.path.join(cfg.data.output, "description", legacy_name)

    has_new = os.path.exists(img_new) or os.path.exists(desc_new)
    has_legacy = os.path.exists(img_legacy) or os.path.exists(desc_legacy)

    if not has_new and has_legacy:
        print(f"  [Paths] No '{run_name}' folder; falling back to legacy folder '{legacy_name}'.")
        return {'output_dir': img_legacy, 'desc_output_dir': desc_legacy}

    return {'output_dir': img_new, 'desc_output_dir': desc_new}