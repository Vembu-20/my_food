'''
# R-CNN

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# FCN for semantic segmentation (COCO-Stuff classes)
fcn_weights = torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
fcn_model = torchvision.models.segmentation.fcn_resnet50(weights=fcn_weights)
fcn_model.eval()

# Faster R-CNN for object detection (COCO)
det_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=det_weights)
det_model.eval()

# Optional: move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fcn_model.to(device)
det_model.to(device)

# Get the right transforms from weights
fcn_transform = fcn_weights.transforms()
det_transform = det_weights.transforms()

# COCO category names (for detection)
coco_categories = det_weights.meta["categories"]  # list of class names

def process_image(image: Image.Image, model_fcn, model_frcnn, score_thresh=0.5):

    # Build two tensors from the same image using their respective transforms
    img_fcn = fcn_transform(image).to(device)
    img_det = det_transform(image).to(device)

    # 3) Inference
    with torch.no_grad():
        # FCN expects batch dimension
        out_fcn = model_fcn(img_fcn.unsqueeze(0))
        seg_logits = out_fcn['out']
        seg_mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()

    with torch.no_grad():
        out_det = model_frcnn([img_det])
    boxes = out_det[0]['boxes'].cpu().numpy()
    labels = out_det[0]['labels'].cpu().numpy()
    scores = out_det[0]['scores'].cpu().numpy()

    # Apply score threshold
    keep = scores >= score_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # 4) Visualization
    cmap = plt.get_cmap('tab20')
    seg_color = cmap((seg_mask % 20) / 20.0)  # (H, W, 4) RGBA

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Segmentation (overlay)
    ax[1].imshow(image)
    ax[1].imshow(seg_color, alpha=0.5)
    ax[1].set_title('FCN Segmentation (overlay)')
    ax[1].axis('off')

    # Detection
    ax[2].imshow(image)
    for (x_min, y_min, x_max, y_max), lab, sc in zip(boxes, labels, scores):
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             fill=False, color='red', linewidth=2)
        ax[2].add_patch(rect)
        # Label text
        class_name = coco_categories[lab] if lab < len(coco_categories) else str(lab)
        ax[2].text(x_min, y_min - 3, f'{class_name} {sc:.2f}',
                   fontsize=9, color='yellow', bbox=dict(facecolor='black', alpha=0.5, pad=1))
    ax[2].set_title('Faster R-CNN Detection')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

# 5) Load image and run
test_image1 = Image.open('/content/1699542504_assignment2_picture.png').convert('RGB')

process_image(test_image1, fcn_model, det_model)
