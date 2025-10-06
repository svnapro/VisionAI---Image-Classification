# imports

import io
import zipfile
import torch
import torch.nn.functional as F
from torchvision import transforms as T

# ---- ImageNet Labels (downloaded with torchvision) ----
from torchvision.models import ResNet50_Weights
IMAGENET_CLASSES = ResNet50_Weights.DEFAULT.meta["categories"]

# Transform for Mask R-CNN
def pil_to_tensor(image):
    return T.ToTensor()(image)

# Transform for classifier (ResNet)
classifier_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def refine_with_classifier(pil_crop, classifier, device="cpu"):
    """Classify cropped object with ResNet50 (ImageNet)."""
    x = classifier_tf(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(x)
        probs = F.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    return IMAGENET_CLASSES[idx.item()], float(conf.item())

def make_zip_bytes(image_bytes_list, filenames):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for data, name in zip(image_bytes_list, filenames):
            z.writestr(name, data.getvalue() if hasattr(data, "getvalue") else data)
    buf.seek(0)
    return buf
