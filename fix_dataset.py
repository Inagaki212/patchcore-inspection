import os
import glob
from PIL import Image

base_path = "./mvtec_anomaly_detection/bottle"
test_dir = os.path.join(base_path, "test")
gt_dir = os.path.join(base_path, "ground_truth")

for folder in os.listdir(test_dir):
    cur_test_path = os.path.join(test_dir, folder)
    cur_gt_path = os.path.join(gt_dir, folder)
    
    if not os.path.isdir(cur_test_path): continue
    os.makedirs(cur_gt_path, exist_ok=True)
    
    test_imgs = sorted(glob.glob(os.path.join(cur_test_path, "*.png")))
    for img_path in test_imgs:
        filename = os.path.basename(img_path)
        # MVTec形式ではマスク名は "000_mask.png" または "000.png"
        # このプログラムの仕様に合わせて両方のパターンで確認し、なければ作成します
        mask_path = os.path.join(cur_gt_path, filename)
        
        if not os.path.exists(mask_path):
            img = Image.open(img_path)
            # 真っ黒な画像（マスク）を作成
            black_mask = Image.new("L", img.size, 0)
            black_mask.save(mask_path)
            print(f"Created mask for: {folder}/{filename}")

print("Dataset fix completed!")
