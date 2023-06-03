import torch
import torchvision
from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from typing import List, Union

class DinoV2Model(torch.nn.Module):
    def __init__(self, device, ts=0):
        super().__init__()
        self.device = device
        self.ts =  ts
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        self.encoder.to(self.device)

    def preprocess(self, x):
        from torchvision.transforms import Resize
        try:
            from torchvision.transforms import InterpolationMode
            BICUBIC = InterpolationMode.BICUBIC
        except ImportError:
            BICUBIC = Image.BICUBIC

        new_width = (63) * 14
        new_height = (84) * 14

        def _to_rgb(x):
            if x.mode != "RGB":
                x = x.convert("RGB")
            return x
        
        return torchvision.transforms.Compose([
            Resize((new_height, new_width), interpolation=BICUBIC),
            _to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(x), new_width, new_height
    
    def forward(self, x):
        with torch.no_grad():
            image = np.array(x)

            preprocessed_image, w, h = self.preprocess(Image.fromarray(image))
            preprocessed_image = preprocessed_image.unsqueeze(0).to(self.device)
            features = self.encoder.forward_features(preprocessed_image)
            feature_grid = features["x_norm_patchtokens"].view(1, h//14, w//14, -1)
        return feature_grid.detach().cpu().numpy()[0, :, :, :]

