import torch
import torchvision
from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from typing import List, Union

class SAMDinoV2Model(torch.nn.Module):
    def __init__(self, device, ts=0):
        super().__init__()
        self.device = device
        self.ts =  ts
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
        
        self.encoder.to(self.device)
        self.sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def preprocess(self, x):
        width, height = x.size
        new_width = (width // 14) * 14
        new_height = (height // 14) * 14

        def _to_rgb(x):
            if x.mode != "RGB":
                x = x.convert("RGB")
            return x
        
        return torchvision.transforms.Compose([
            _to_rgb,
            torchvision.transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(x)
    
    def forward(self, x):
        original_width, original_height = x.size

        with torch.no_grad():
            masks = self.mask_generator.generate(np.array(x))
            masks = self.filter_masks(masks)
            mask = self.merge_masks(masks)
            foreground_image = np.array(x) * mask[..., None]

            preprocessed_image = self.preprocess(Image.fromarray(foreground_image)).unsqueeze(0).to(self.device)
            features = self.encoder.forward_features(preprocessed_image)
            h, w = original_height // 14, original_width // 14
            feature_grid = features["x_norm_patchtokens"].view(1, h, w, -1)

            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()

            mask_rescaled = cv2.resize(mask.astype(float), (w, h))

            feature_grid = feature_grid * torch.from_numpy(mask_rescaled)[None, ..., None].to(self.device)
            pca_grid = self.compute_first_3_pca(feature_grid)
            

        return pca_grid
    
    def filter_masks(self, masks):
        filtered_masks = []
        for mask in masks:
            if 0.25 > np.count_nonzero(mask['segmentation']) / mask['segmentation'].size > 0.01:
                filtered_masks.append(mask)
        return filtered_masks
    
    def merge_masks(self, masks):
        merged_mask = np.zeros(masks[0]['segmentation'].shape)
        for mask in masks:
            merged_mask = np.logical_or(merged_mask, mask['segmentation'])
        return merged_mask

    def compute_first_3_pca(self, input_tensors: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:

        if isinstance(input_tensors, torch.Tensor):
            input_tensors = [input_tensors]

        output_tensors = []

        concatenated_tensors = []
        for tensor in input_tensors:
            if len(tensor.shape) != 4:
                raise ValueError("Input tensors should have 4 dimensions: (1, H, W, N)")

            nonzero_indices = (tensor[0] != 0).any(dim=-1)
            flattened_nonzero = tensor[0, nonzero_indices].view(-1, tensor.shape[3])
            concatenated_tensors.append(flattened_nonzero)
        concatenated_tensors = torch.cat(concatenated_tensors, dim=0)

        _, _, V = torch.pca_lowrank(concatenated_tensors, center=True)

        first_3_components = V[:, :3]
        projected_tensors = concatenated_tensors @ first_3_components

        per_channel_mins = [projected_tensors[:, i].min() for i in range(projected_tensors.shape[-1])]
        per_channel_maxes = [projected_tensors[:, i].max() - per_channel_mins[i] for i in range(projected_tensors.shape[-1])]

        for tensor in input_tensors:
            H, W, N = tensor.shape[1:]
            reshaped_projected_tensor = (tensor @ first_3_components).view(1, H, W, 3)
            for i in range(3):
                reshaped_projected_tensor[:, :, :, i][reshaped_projected_tensor[:, :, :, i] != 0] -= per_channel_mins[i]
                reshaped_projected_tensor[:, :, :, i][reshaped_projected_tensor[:, :, :, i] != 0] /= per_channel_maxes[i]

            output_tensors.append(reshaped_projected_tensor)

        return output_tensors

def preprocess_rgb(img):

    from torchvision.transforms import Compose, Resize, CenterCrop
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    preprocess = Compose([
        Resize(800, interpolation=BICUBIC),
        _convert_image_to_rgb,
    ])

    return preprocess(img)

