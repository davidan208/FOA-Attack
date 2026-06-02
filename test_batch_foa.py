"""
Test script to verify FOA Attack batch > 1 capability.

This script tests whether the EnsembleFeatureExtractor_ot and 
EnsembleFeatureLoss_OT_foa_attack can handle batch_size > 1 correctly.

Uses MPS (Metal Performance Shaders) backend for macOS.
"""
import os
import sys
import torch
import numpy as np

# Determine device - use MPS if available on macOS
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


def test_feature_extractor_batch():
    """Test EnsembleFeatureExtractor_ot with batch_size > 1."""
    from surrogates import (
        ClipB16FeatureExtractor,
        ClipB32FeatureExtractor,
        ClipLaionFeatureExtractor,
        EnsembleFeatureExtractor_ot,
    )

    print("\n" + "=" * 60)
    print("TEST 1: EnsembleFeatureExtractor_ot with batch_size > 1")
    print("=" * 60)

    # Create models
    models = []
    for ModelClass in [ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor]:
        model = ModelClass().eval().to(DEVICE).requires_grad_(False)
        models.append(model)

    ensemble_extractor = EnsembleFeatureExtractor_ot(models, cluster_number=3)

    # Test with batch_size=1 (baseline)
    print("\n--- batch_size=1 (baseline) ---")
    x_single = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    try:
        features_single, features_local_single = ensemble_extractor(x_single)
        print(f"  Global features keys: {list(features_single.keys())}")
        for k, v in features_single.items():
            print(f"    Model {k}: shape={v.shape}")
        for k, v in features_local_single.items():
            print(f"    Model {k} local: shape={v.shape}")
        print("  ✅ batch_size=1 PASSED")
    except Exception as e:
        print(f"  ❌ batch_size=1 FAILED: {e}")
        return False

    # Test with batch_size=2
    print("\n--- batch_size=2 ---")
    x_batch2 = torch.randn(2, 3, 224, 224).to(DEVICE) * 255.0
    try:
        features_batch2, features_local_batch2 = ensemble_extractor(x_batch2)
        print(f"  Global features keys: {list(features_batch2.keys())}")
        for k, v in features_batch2.items():
            print(f"    Model {k}: shape={v.shape}")
        for k, v in features_local_batch2.items():
            print(f"    Model {k} local: shape={v.shape}")
        print("  ✅ batch_size=2 PASSED")
    except Exception as e:
        print(f"  ❌ batch_size=2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with batch_size=4
    print("\n--- batch_size=4 ---")
    x_batch4 = torch.randn(4, 3, 224, 224).to(DEVICE) * 255.0
    try:
        features_batch4, features_local_batch4 = ensemble_extractor(x_batch4)
        print(f"  Global features keys: {list(features_batch4.keys())}")
        for k, v in features_batch4.items():
            print(f"    Model {k}: shape={v.shape}")
        for k, v in features_local_batch4.items():
            print(f"    Model {k} local: shape={v.shape}")
        print("  ✅ batch_size=4 PASSED")
    except Exception as e:
        print(f"  ❌ batch_size=4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_loss_function_batch():
    """Test EnsembleFeatureLoss_OT_foa_attack with batch_size > 1."""
    from surrogates import (
        ClipB16FeatureExtractor,
        ClipB32FeatureExtractor,
        ClipLaionFeatureExtractor,
        EnsembleFeatureExtractor_ot,
        EnsembleFeatureLoss_OT_foa_attack,
    )

    print("\n" + "=" * 60)
    print("TEST 2: EnsembleFeatureLoss_OT_foa_attack with batch_size > 1")
    print("=" * 60)

    # Create models
    models = []
    for ModelClass in [ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor]:
        model = ModelClass().eval().to(DEVICE).requires_grad_(False)
        models.append(model)

    ensemble_extractor = EnsembleFeatureExtractor_ot(models, cluster_number=3)
    ensemble_loss = EnsembleFeatureLoss_OT_foa_attack(models, cluster_number=3)

    # Test with batch_size=1 (baseline)
    print("\n--- batch_size=1 loss (baseline) ---")
    x_src = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    x_tgt = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    try:
        # Set ground truth
        ensemble_loss.set_ground_truth(x_tgt)
        # Extract features
        features, features_local = ensemble_extractor(x_src)
        # Compute loss
        loss = ensemble_loss(features, features_local)
        print(f"  Loss value: {loss.item():.6f}")
        print("  ✅ batch_size=1 loss PASSED")
    except Exception as e:
        print(f"  ❌ batch_size=1 loss FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with batch_size=2
    print("\n--- batch_size=2 loss ---")
    x_src_b2 = torch.randn(2, 3, 224, 224).to(DEVICE) * 255.0
    x_tgt_b2 = torch.randn(2, 3, 224, 224).to(DEVICE) * 255.0
    try:
        ensemble_loss.previous_loss_list = []  # Reset
        ensemble_loss.set_ground_truth(x_tgt_b2)
        features_b2, features_local_b2 = ensemble_extractor(x_src_b2)
        loss_b2 = ensemble_loss(features_b2, features_local_b2)
        print(f"  Loss value: {loss_b2.item():.6f}")
        print("  ✅ batch_size=2 loss PASSED")
    except Exception as e:
        print(f"  ❌ batch_size=2 loss FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_fgsm_attack_batch():
    """Test the full fgsm_attack function with batch_size > 1."""
    from surrogates import (
        ClipB16FeatureExtractor,
        ClipB32FeatureExtractor,
        ClipLaionFeatureExtractor,
        EnsembleFeatureExtractor_ot,
        EnsembleFeatureLoss_OT_foa_attack,
    )
    from torchvision import transforms
    import wandb

    print("\n" + "=" * 60)
    print("TEST 3: Full fgsm_attack with batch_size > 1")
    print("=" * 60)

    # Initialize wandb in offline mode for testing
    wandb.init(mode="disabled")

    # Create models
    models = []
    for ModelClass in [ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor]:
        model = ModelClass().eval().to(DEVICE).requires_grad_(False)
        models.append(model)

    ensemble_extractor = EnsembleFeatureExtractor_ot(models, cluster_number=3)
    ensemble_loss = EnsembleFeatureLoss_OT_foa_attack(models, cluster_number=3)

    # Import the attack function
    from generate_adversarial_samples_foa_attack import fgsm_attack
    from config_schema import MainConfig, DataConfig, OptimConfig, ModelConfig

    # Create a minimal config
    cfg = MainConfig()
    cfg.model.device = DEVICE
    cfg.model.input_res = 224
    cfg.model.use_source_crop = False
    cfg.model.use_target_crop = False
    cfg.optim.steps = 5  # Very few steps for testing
    cfg.optim.alpha = 1.0
    cfg.optim.epsilon = 8

    source_crop = torch.nn.Identity()
    target_crop = torch.nn.Identity()

    # Test batch_size=1
    print("\n--- fgsm_attack batch_size=1 ---")
    x_src = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    x_tgt = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    try:
        ensemble_loss.previous_loss_list = []
        adv = fgsm_attack(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=0,
            image_org=x_src,
            image_tgt=x_tgt,
        )
        print(f"  Output shape: {adv.shape}")
        print("  ✅ fgsm batch_size=1 PASSED")
    except Exception as e:
        print(f"  ❌ fgsm batch_size=1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test batch_size=2
    print("\n--- fgsm_attack batch_size=2 ---")
    x_src_b2 = torch.randn(2, 3, 224, 224).to(DEVICE) * 255.0
    x_tgt_b2 = torch.randn(2, 3, 224, 224).to(DEVICE) * 255.0
    try:
        ensemble_loss.previous_loss_list = []
        adv_b2 = fgsm_attack(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=0,
            image_org=x_src_b2,
            image_tgt=x_tgt_b2,
        )
        print(f"  Output shape: {adv_b2.shape}")
        print("  ✅ fgsm batch_size=2 PASSED")
    except Exception as e:
        print(f"  ❌ fgsm batch_size=2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test batch_size=4
    print("\n--- fgsm_attack batch_size=4 ---")
    x_src_b4 = torch.randn(4, 3, 224, 224).to(DEVICE) * 255.0
    x_tgt_b4 = torch.randn(4, 3, 224, 224).to(DEVICE) * 255.0
    try:
        ensemble_loss.previous_loss_list = []
        adv_b4 = fgsm_attack(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=0,
            image_org=x_src_b4,
            image_tgt=x_tgt_b4,
        )
        print(f"  Output shape: {adv_b4.shape}")
        print("  ✅ fgsm batch_size=4 PASSED")
    except Exception as e:
        print(f"  ❌ fgsm batch_size=4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    wandb.finish()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("FOA Attack Batch Size > 1 Capability Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    results = {}

    # Test 1: Feature extractor
    results["feature_extractor"] = test_feature_extractor_batch()

    # Test 2: Loss function
    results["loss_function"] = test_loss_function_batch()

    # Test 3: Full attack
    results["fgsm_attack"] = test_fgsm_attack_batch()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
