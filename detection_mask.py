import sys, os, cv2, pathlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.transforms import functional as F

img_fld_path = pathlib.Path(sys.argv[1])
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
img_paths = sorted([p for p in img_fld_path.iterdir() if p.suffix in IMG_EXTS])
out_dir_path = pathlib.Path(sys.argv[2])
if not os.path.exists(out_dir_path): os.mkdir(out_dir_path)

# FCOSモデルの初期化
device = "cuda" if torch.cuda.is_available() else "cpu"
model = fcos_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

for i in range(len(img_paths)):
    # print(img_paths[i])
    # カラー画像として読み込み
    img = Image.open(str(img_paths[i])).convert('RGB') #読み込み
    print(img_paths[i])

    # 物体検出のための前処理
    img_tensor = F.to_tensor(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 物体検出の実行
    with torch.no_grad():
        detection_results = model(img_tensor)

    # 検出結果の取得
    boxes = detection_results[0]['boxes'].cpu().numpy()
    scores = detection_results[0]['scores'].cpu().numpy()
    labels = detection_results[0]["labels"].detach().cpu().numpy()

    draw = ImageDraw.Draw(img)
    for i in range(len(scores)):
        prd_val = scores[i]
        # 信頼度スコアが高い検出結果のみを使用
        threshold = 0.5
        if prd_val < threshold: break # 閾値以下が出現した段階で終了

        x1, y1, x2, y2 = boxes[i]
        print(x1, y1, x2, y2)
        
        left_posi = (int(x1), int(y1))
        right_posi = (int(x2), int(y2))

        box_col = (255, 0, 0)
        linewidth = 3
        draw.rectangle((left_posi, right_posi), outline=box_col, width=linewidth)

    # 検出結果を描画した画像を保存
    output_path = out_dir_path / f"{img_paths[i].stem}_detected.png"
    img.save(output_path)