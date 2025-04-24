#ピクセルが画像全体の平均値以下のときにマスク処理をする
import sys, os, cv2, pathlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

img_fld_path = pathlib.Path(sys.argv[1])
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
img_paths = sorted([p for p in img_fld_path.iterdir() if p.suffix in IMG_EXTS])
out_dir_path = pathlib.Path(sys.argv[2])
if not os.path.exists(out_dir_path): os.mkdir(out_dir_path)

for i in range(len(img_paths)):
    # print(img_paths[i])
    # カラー画像として読み込み
    img = Image.open(str(img_paths[i])).convert('RGB') #読み込み

    # グレースケールに変換
    img_gray_pil = img.convert('L')
    
    # PIL ImageをNumPy配列に変換
    img_gray_np = np.array(img_gray_pil)
    
    # 画像の平均値を計算
    mean_value = np.mean(img_gray_np)
    print(f"Mean value: {mean_value}")

    # 平均値以下のピクセルをマスク（黒に設定）
    img_gray_np[img_gray_np <= mean_value] = 0

    # マスクを3チャンネルに拡張
    mask_3channel = np.stack([img_gray_np] * 3, axis=-1)
    
    # 元のカラー画像にマスクを適用
    result_img = np.where(mask_3channel == 0, 0, np.array(img))

    # 結果をPIL Imageに変換して保存
    result_img_pil = Image.fromarray(result_img.astype(np.uint8))
    output_path = out_dir_path / f"{img_paths[i].stem}.png"
    result_img_pil.save(output_path)