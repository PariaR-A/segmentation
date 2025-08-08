import torch
import torch.nn as nn
from torchvision.ops import box_iou, masks_to_boxes

class SwinUnetDetection(nn.Module):
    def __init__(self, swin_unet, threshold=0.5):
        super().__init__()
        self.swin_unet = swin_unet
        self.threshold = threshold

    def forward(self, x, target_masks=None):
        """
        Training: returns dict(loss=...)
        Testing: returns list of boxes per image
        """
        seg_logits = self.swin_unet(x)  # (B, C, H, W)
        seg_probs = torch.sigmoid(seg_logits)

        if self.training:
            assert target_masks is not None, "Need masks in training"
            gt_boxes = [masks_to_boxes((m > 0).float()) for m in target_masks]
            pred_boxes = [masks_to_boxes((p > self.threshold).float()) for p in seg_probs]

            loss = 0.0
            for pb, gb in zip(pred_boxes, gt_boxes):
                if pb.numel() == 0 or gb.numel() == 0:
                    continue
                iou = box_iou(pb, gb)
                loss += (1 - iou).mean()

            return {"loss": loss}

        else:
            detections = [masks_to_boxes((p > self.threshold).float()) for p in seg_probs]
            return detections
