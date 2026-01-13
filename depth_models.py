import os
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config_loader import get_depth_model_id, get_unet_model_path, get_unet_use_batchnorm


def _get_device(device: Optional[str] = None) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth_min = float(np.nanmin(depth))
    depth_max = float(np.nanmax(depth))
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    return depth_norm.astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = False) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if use_batchnorm:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        if use_batchnorm:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
                nn.ReLU(inplace=True),
            ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Tuple[int, ...] = (64, 128, 256, 512),
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature, use_batchnorm))
            in_ch = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, use_batchnorm)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, use_batchnorm))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class UNetDConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Tuple[int, ...] = (64, 128, 256, 512),
        use_batchnorm: bool = True,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        if len(features) != 4:
            raise ValueError("UNetDConv requires 4 feature sizes.")
        f1, f2, f3, f4 = features
        self.dconv_down1 = DoubleConvBlock(in_channels, f1, use_batchnorm, conv_bias)
        self.dconv_down2 = DoubleConvBlock(f1, f2, use_batchnorm, conv_bias)
        self.dconv_down3 = DoubleConvBlock(f2, f3, use_batchnorm, conv_bias)
        self.dconv_down4 = DoubleConvBlock(f3, f4, use_batchnorm, conv_bias)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up3 = DoubleConvBlock(f4 + f3, f3, use_batchnorm, conv_bias)
        self.dconv_up2 = DoubleConvBlock(f3 + f2, f2, use_batchnorm, conv_bias)
        self.dconv_up1 = DoubleConvBlock(f2 + f1, f1, use_batchnorm, conv_bias)
        self.conv_last = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        return self.conv_last(x)


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _state_dict_has_bn(state) -> bool:
    return any("running_mean" in k or "running_var" in k for k in state.keys())


def _state_dict_is_dconv(state) -> bool:
    return any(k.startswith("dconv_down1.") for k in state.keys())


def _state_dict_has_conv_bias(state) -> bool:
    for key in state.keys():
        if key.endswith(".double_conv.0.bias") or key.endswith(".double_conv.3.bias"):
            return True
    return False


def _infer_dconv_features(state) -> Tuple[Tuple[int, int, int, int], int, int]:
    def get_out(key, default):
        weight = state.get(key)
        if weight is None:
            return default
        return int(weight.shape[0])

    def get_in(key, default):
        weight = state.get(key)
        if weight is None:
            return default
        return int(weight.shape[1])

    f1 = get_out("dconv_down1.double_conv.0.weight", 64)
    f2 = get_out("dconv_down2.double_conv.0.weight", 128)
    f3 = get_out("dconv_down3.double_conv.0.weight", 256)
    f4 = get_out("dconv_down4.double_conv.0.weight", 512)
    in_ch = get_in("dconv_down1.double_conv.0.weight", 3)
    out_ch = get_out("conv_last.weight", 1)
    return (f1, f2, f3, f4), in_ch, out_ch


@lru_cache(maxsize=2)
def _load_depth_anything(model_id: str, device_type: str):
    device = torch.device(device_type)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=2)
def _load_unet(model_path: str, device_type: str, use_batchnorm: bool):
    device = torch.device(device_type)
    state = torch.load(model_path, map_location=device)
    state = _unwrap_state_dict(state)
    auto_bn = _state_dict_has_bn(state)
    auto_bias = _state_dict_has_conv_bias(state)
    if _state_dict_is_dconv(state):
        features, in_ch, out_ch = _infer_dconv_features(state)
        model = UNetDConv(
            in_channels=in_ch,
            out_channels=out_ch,
            features=features,
            use_batchnorm=auto_bn,
            conv_bias=auto_bias,
        ).to(device)
    else:
        if auto_bn and not use_batchnorm:
            use_batchnorm = True
        model = UNet(in_channels=3, out_channels=1, use_batchnorm=use_batchnorm).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_depth_anything(
    image_path: str,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model_id = model_id or get_depth_model_id()
    device_t = _get_device(device)
    processor, model = _load_depth_anything(model_id, device_t.type)

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device_t)

    with torch.no_grad():
        depth = model(**inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth_np = depth.squeeze().cpu().numpy()
    depth_norm = _normalize_depth(depth_np)

    sky_threshold = 0.03
    depth_norm[depth_norm < sky_threshold] = 0.0

    return depth_norm, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def predict_depth_unet(
    image_path: str,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    input_size: Tuple[int, int] = (240, 320),
    use_batchnorm: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model_path = (model_path or get_unet_model_path()).strip()
    if not model_path:
        raise FileNotFoundError("UNet model path is empty.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"UNet model not found: {model_path}")

    device_t = _get_device(device)
    if use_batchnorm is None:
        use_batchnorm = get_unet_use_batchnorm()
    model = _load_unet(os.path.abspath(model_path), device_t.type, bool(use_batchnorm))

    original_img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )
    input_tensor = preprocess(original_img).unsqueeze(0).to(device_t)

    with torch.no_grad():
        output = model(input_tensor)

    depth_map = output.squeeze().cpu().numpy()
    if depth_map.ndim != 2:
        depth_map = depth_map.squeeze()

    orig_w, orig_h = original_img.size
    depth_resized = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    depth_norm = _normalize_depth(depth_resized)

    return depth_norm, cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)


def predict_depth_map(
    image_path: str,
    model_choice: Optional[str] = None,
    model_id: Optional[str] = None,
    unet_model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    choice = (model_choice or "depth-anything").strip().lower()
    if choice in ("unet", "custom", "local"):
        return predict_depth_unet(image_path, unet_model_path, device=device)
    return predict_depth_anything(image_path, model_id=model_id, device=device)


def render_depth_colormap(depth_norm: np.ndarray) -> np.ndarray:
    depth_uint8 = np.clip(depth_norm, 0.0, 1.0) * 255.0
    depth_uint8 = depth_uint8.astype(np.uint8)
    cmap = cv2.COLORMAP_MAGMA if hasattr(cv2, "COLORMAP_MAGMA") else cv2.COLORMAP_INFERNO
    return cv2.applyColorMap(depth_uint8, cmap)
