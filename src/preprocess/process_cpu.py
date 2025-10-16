# import os, torch
# from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
# import torchvision.transforms as T
# from PIL import Image
# dataset_path = os.path.expanduser('/Users/dario/Desktop/uni/1_TESI/GazeXplain/GazeXplain/dataset_root/OSIE')
# device = torch.device('cpu')
# backbone = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device).eval()
# resize = T.Resize((384*2, 512*2))
# normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# for fname in os.listdir(os.path.join(dataset_path, "stimuli")):
#     if not fname.endswith(".jpg"): 
#         continue
#     img = Image.open(os.path.join(dataset_path, "stimuli", fname))
#     tensor = normalize(resize(T.functional.to_tensor(img))).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feat = backbone(tensor)
#     x = feat['pool'].squeeze().cpu() if isinstance(feat, dict) and 'pool' in feat else feat['0'].squeeze().cpu()
#     torch.save(x, os.path.join(dataset_path, "image_features", fname.replace(".jpg", ".pth")))
    
    
import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image

class ResNetCOCO(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.resnet = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        b, c, h, w = x.shape
        return x.view(b, c, h * w).permute(0, 2, 1).contiguous()

dataset_path = os.path.expanduser('/Users/dario/Desktop/uni/1_TESI/GazeXplain/GazeXplain/dataset_root/OSIE')
stimuli_dir = os.path.join(dataset_path, 'stimuli')
target_dir = os.path.join(dataset_path, 'image_features')
os.makedirs(target_dir, exist_ok=True)

resize = T.Resize((384 * 2, 512 * 2))
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
backbone = ResNetCOCO(device="cpu").eval()

for fname in sorted(os.listdir(stimuli_dir)):
    if not fname.lower().endswith('.jpg'):
        continue
    image = Image.open(os.path.join(stimuli_dir, fname)).convert('RGB')
    tensor = normalize(resize(T.functional.to_tensor(image))).unsqueeze(0)
    features = backbone(tensor).squeeze(0).cpu()
    torch.save(features, os.path.join(target_dir, fname.replace('.jpg', '.pth')))