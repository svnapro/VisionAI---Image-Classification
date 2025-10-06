import torchvision.models.detection as detection
import torchvision.models as models

def get_maskrcnn(device=None):
    """Load pretrained Mask R-CNN (COCO)."""
    model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    if device: model.to(device)
    return model

def get_resnet50(device=None):
    """Load pretrained ResNet50 (ImageNet)."""
    model = models.resnet50(pretrained=True)
    model.eval()
    if device: model.to(device)
    return model
