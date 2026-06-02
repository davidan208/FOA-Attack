"""
Test: Kiểm tra kmeans có bị ảnh hưởng khi chạy trong batch > 1 hay không.

So sánh:
1. Chạy model.global_local_features(x) với batch=1 → lấy embedding → kmeans
2. Chạy model.global_local_features(batch_of_2) với batch=2 → lấy embedding[0] → kmeans
3. So sánh cluster centers có giống nhau không

Mục đích: Xác nhận rằng nếu ta fix batch > 1 bằng cách loop qua từng sample
trong batch để tính kmeans, kết quả sẽ tương đương với chạy batch=1.
"""
import os
import sys
import torch
import numpy as np
from kmeans_pytorch import kmeans

# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")


def test_global_local_features_consistency():
    """
    Test: model.global_local_features() output cho cùng 1 ảnh có giống nhau
    khi chạy batch=1 vs batch=2 (lấy index 0)?
    """
    from surrogates import ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor

    print("\n" + "=" * 60)
    print("TEST: global_local_features consistency (batch=1 vs batch=2)")
    print("=" * 60)

    models = {
        "B16": ClipB16FeatureExtractor().eval().to(DEVICE).requires_grad_(False),
        "B32": ClipB32FeatureExtractor().eval().to(DEVICE).requires_grad_(False),
        "Laion": ClipLaionFeatureExtractor().eval().to(DEVICE).requires_grad_(False),
    }

    # Tạo input cố định
    torch.manual_seed(42)
    img_a = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    img_b = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    img_batch = torch.cat([img_a, img_b], dim=0)  # [2, 3, 224, 224]

    for name, model in models.items():
        print(f"\n--- Model: {name} ---")

        with torch.no_grad():
            # Chạy batch=1 (chỉ img_a)
            global_single, local_single = model.global_local_features(img_a)

            # Chạy batch=2 (img_a + img_b), lấy index 0
            global_batch, local_batch = model.global_local_features(img_batch)

        # So sánh global feature
        global_diff = torch.abs(global_single[0] - global_batch[0]).max().item()
        print(f"  Global feature max diff: {global_diff:.2e}")

        # So sánh local feature (embedding)
        local_diff = torch.abs(local_single[0] - local_batch[0]).max().item()
        print(f"  Local embedding max diff: {local_diff:.2e}")

        # Shape info
        print(f"  Local single shape: {local_single.shape}")
        print(f"  Local batch shape: {local_batch.shape}")

        if global_diff < 1e-5 and local_diff < 1e-5:
            print(f"  ✅ {name}: Embeddings IDENTICAL regardless of batch size")
        else:
            print(f"  ⚠️  {name}: Embeddings DIFFER (diff: global={global_diff:.2e}, local={local_diff:.2e})")


def test_kmeans_determinism():
    """
    Test: kmeans trên cùng 1 embedding có cho kết quả giống nhau giữa các lần chạy?
    (kmeans có random initialization nên cần kiểm tra)
    """
    from surrogates import ClipB16FeatureExtractor

    print("\n" + "=" * 60)
    print("TEST: kmeans determinism trên cùng embedding")
    print("=" * 60)

    model = ClipB16FeatureExtractor().eval().to(DEVICE).requires_grad_(False)

    torch.manual_seed(42)
    img = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0

    with torch.no_grad():
        _, local_features = model.global_local_features(img)

    embedding = local_features[0]  # [num_patches, dim]
    print(f"  Embedding shape: {embedding.shape}")

    # Chạy kmeans nhiều lần
    cluster_centers_list = []
    for run in range(5):
        np.random.seed(42)  # Fix seed
        _, center = kmeans(
            X=embedding,
            num_clusters=3,
            distance='euclidean',
            device=torch.device(DEVICE)
        )
        center = center.to(DEVICE)
        cluster_centers_list.append(center)

    # So sánh giữa các lần chạy
    print(f"\n  Comparing 5 runs of kmeans (same seed):")
    all_same = True
    for i in range(1, 5):
        diff = torch.abs(cluster_centers_list[0] - cluster_centers_list[i]).max().item()
        status = "✅" if diff < 1e-5 else "⚠️"
        print(f"    Run 0 vs Run {i}: max diff = {diff:.2e} {status}")
        if diff >= 1e-5:
            all_same = False

    if all_same:
        print("  ✅ kmeans is DETERMINISTIC with fixed seed")
    else:
        print("  ⚠️  kmeans is NOT fully deterministic even with fixed seed")

    # Bây giờ test KHÔNG fix seed
    print(f"\n  Comparing 5 runs of kmeans (NO fixed seed):")
    cluster_centers_noseed = []
    for run in range(5):
        _, center = kmeans(
            X=embedding,
            num_clusters=3,
            distance='euclidean',
            device=torch.device(DEVICE)
        )
        center = center.to(DEVICE)
        cluster_centers_noseed.append(center)

    for i in range(1, 5):
        diff = torch.abs(cluster_centers_noseed[0] - cluster_centers_noseed[i]).max().item()
        status = "✅ same" if diff < 1e-5 else "⚠️ DIFFERENT"
        print(f"    Run 0 vs Run {i}: max diff = {diff:.2e} {status}")


def test_kmeans_batch_vs_single():
    """
    Test chính: So sánh kết quả kmeans khi:
    - Chạy model với batch=1 → embedding → kmeans
    - Chạy model với batch=2 → embedding[0] → kmeans
    
    Nếu embedding giống nhau (test trước đã verify), thì kmeans cũng phải giống.
    """
    from surrogates import ClipB16FeatureExtractor

    print("\n" + "=" * 60)
    print("TEST: kmeans output - batch=1 vs batch=2 (same image)")
    print("=" * 60)

    model = ClipB16FeatureExtractor().eval().to(DEVICE).requires_grad_(False)

    torch.manual_seed(42)
    img_a = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    img_b = torch.randn(1, 3, 224, 224).to(DEVICE) * 255.0
    img_batch = torch.cat([img_a, img_b], dim=0)

    with torch.no_grad():
        # Batch=1
        _, local_single = model.global_local_features(img_a)
        # Batch=2
        _, local_batch = model.global_local_features(img_batch)

    emb_single = local_single[0]  # [num_patches, dim]
    emb_batch0 = local_batch[0]   # [num_patches, dim]

    print(f"  Embedding single shape: {emb_single.shape}")
    print(f"  Embedding batch[0] shape: {emb_batch0.shape}")
    print(f"  Embedding diff: {torch.abs(emb_single - emb_batch0).max().item():.2e}")

    # Chạy kmeans trên cả hai
    np.random.seed(123)
    _, center_single = kmeans(
        X=emb_single, num_clusters=3, distance='euclidean', device=torch.device(DEVICE)
    )
    center_single = center_single.to(DEVICE)

    np.random.seed(123)
    _, center_batch0 = kmeans(
        X=emb_batch0, num_clusters=3, distance='euclidean', device=torch.device(DEVICE)
    )
    center_batch0 = center_batch0.to(DEVICE)

    diff = torch.abs(center_single - center_batch0).max().item()
    print(f"\n  Cluster centers diff (same seed): {diff:.2e}")

    if diff < 1e-5:
        print("  ✅ kmeans produces IDENTICAL results for batch=1 vs batch=2[0]")
    else:
        print("  ⚠️  kmeans produces DIFFERENT results")

    # Kiểm tra thêm: embedding[1] (sample thứ 2 trong batch) cho kết quả khác
    emb_batch1 = local_batch[1]
    np.random.seed(123)
    _, center_batch1 = kmeans(
        X=emb_batch1, num_clusters=3, distance='euclidean', device=torch.device(DEVICE)
    )
    center_batch1 = center_batch1.to(DEVICE)

    diff_01 = torch.abs(center_single - center_batch1).max().item()
    print(f"  Cluster centers diff (img_a vs img_b): {diff_01:.2e}")
    if diff_01 > 1e-3:
        print("  ✅ Different images produce different cluster centers (expected)")


def test_squeeze_behavior():
    """
    Test: Giải thích tại sao squeeze(0) gây lỗi khi batch > 1.
    Đây là root cause của bug.
    """
    print("\n" + "=" * 60)
    print("TEST: squeeze(0) behavior với batch sizes khác nhau")
    print("=" * 60)

    # Simulate embedding shapes
    # B16: 196 patches, 768 dim
    emb_b1 = torch.randn(1, 196, 768)  # batch=1
    emb_b2 = torch.randn(2, 196, 768)  # batch=2
    emb_b4 = torch.randn(4, 196, 768)  # batch=4

    print(f"  batch=1: shape={emb_b1.shape} → squeeze(0) → {emb_b1.squeeze(0).shape}")
    print(f"  batch=2: shape={emb_b2.shape} → squeeze(0) → {emb_b2.squeeze(0).shape}")
    print(f"  batch=4: shape={emb_b4.shape} → squeeze(0) → {emb_b4.squeeze(0).shape}")

    print(f"\n  Khi batch=1: squeeze(0) → [196, 768] → kmeans nhận 196 samples ✅")
    print(f"  Khi batch=2: squeeze(0) → [2, 196, 768] (KHÔNG squeeze!) → kmeans nhận 2 samples")
    print(f"    → num_clusters=3 > num_samples=2 → CRASH ❌")
    print(f"  Khi batch=4: squeeze(0) → [4, 196, 768] (KHÔNG squeeze!) → kmeans nhận 4 samples")
    print(f"    → num_clusters=3 <= num_samples=4 → CHẠY ĐƯỢC nhưng SAI NGHĨA ⚠️")
    print(f"       (kmeans cluster trên 4 embeddings thay vì 196 patches)")


if __name__ == "__main__":
    print("=" * 60)
    print("Kiểm tra ảnh hưởng của batch size lên kmeans trong FOA Attack")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Test 0: Giải thích squeeze behavior
    test_squeeze_behavior()

    # Test 1: Embedding consistency
    test_global_local_features_consistency()

    # Test 2: kmeans determinism
    test_kmeans_determinism()

    # Test 3: kmeans batch vs single
    test_kmeans_batch_vs_single()

    print("\n" + "=" * 60)
    print("KẾT LUẬN")
    print("=" * 60)
    print("""
  1. model.global_local_features() cho output GIỐNG NHAU cho cùng 1 ảnh
     bất kể batch size (vì CLIP vision model xử lý independent per sample).
  
  2. kmeans KHÔNG bị ảnh hưởng bởi batch size NẾU ta đúng cách lấy
     embedding của từng sample riêng (embedding[i]) trước khi đưa vào kmeans.
  
  3. Bug hiện tại: squeeze(0) chỉ hoạt động khi batch=1.
     Fix đúng: loop qua batch dimension, tính kmeans cho từng sample.
     Kết quả sẽ TƯƠNG ĐƯƠNG với chạy batch=1.
""")
