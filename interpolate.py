import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config_loader import get_depth_model_id
from depth_models import predict_depth_map

def _get_device() -> torch.device:
    """Returns the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _get_lama_inpainter(device: Optional[torch.device] = None):
    """
    Lazy loader for the LaMa inpainting model. Raises a clear error if the
    dependency is missing so the caller can surface the installation hint.
    """
    if SimpleLama is None:
        raise ImportError(
            "simple-lama-inpainting is required for LaMa inpainting. "
            "Add it to your environment: pip install simple-lama-inpainting"
        )
    return SimpleLama(device=device or _get_device())


def lama_inpaint(image_bgr: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """
    Inpaints holes using LaMa.

    Args:
        image_bgr: Input image in BGR uint8 format.
        hole_mask: Binary mask (255 on holes) where inpainting is required.
    """
    if image_bgr is None or hole_mask is None:
        raise ValueError("image_bgr and hole_mask must be provided to LaMa.")

    if hole_mask.ndim == 3:
        hole_mask = cv2.cvtColor(hole_mask, cv2.COLOR_BGR2GRAY)
    mask = (hole_mask > 0).astype(np.uint8) * 255

    inpainter = _get_lama_inpainter()
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = inpainter(rgb_image, mask)
    result_np = np.array(result_rgb.convert("RGB"))
    return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)


def _softmax_splat(
    ten_input: torch.Tensor,
    ten_flow: torch.Tensor,
    ten_metric: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lightweight PyTorch implementation of Softmax Splatting.

    Args:
        ten_input: Tensor shaped (B, C, H, W).
        ten_flow: Tensor shaped (B, 2, H, W) giving dx, dy in pixel space.
        ten_metric: Tensor shaped (B, 1, H, W) providing importance weights.

    Returns:
        splatted tensor (B, C, H, W) and the accumulated weights (B, 1, H, W).
    """
    B, C, H, W = ten_input.shape
    device = ten_input.device

    base_x = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    base_y = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

    tgt_x = base_x + ten_flow[:, 0:1]
    tgt_y = base_y + ten_flow[:, 1:2]

    valid = (
        (tgt_x >= 0)
        & (tgt_x <= (W - 1))
        & (tgt_y >= 0)
        & (tgt_y <= (H - 1))
        & torch.isfinite(ten_metric)
    ).to(ten_input.dtype)

    weight = torch.exp(torch.clamp(ten_metric, max=50.0)) * valid

    x0 = torch.floor(tgt_x)
    y0 = torch.floor(tgt_y)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - tgt_x) * (y1 - tgt_y)
    wb = (x1 - tgt_x) * (tgt_y - y0)
    wc = (tgt_x - x0) * (y1 - tgt_y)
    wd = (tgt_x - x0) * (tgt_y - y0)

    x0 = torch.clamp(x0, 0, W - 1).long()
    x1 = torch.clamp(x1, 0, W - 1).long()
    y0 = torch.clamp(y0, 0, H - 1).long()
    y1 = torch.clamp(y1, 0, H - 1).long()

    output = torch.zeros_like(ten_input)
    accum = torch.zeros((B, 1, H, W), device=device, dtype=ten_input.dtype)

    def scatter(weight_part, x_idx, y_idx):
        idx = (y_idx * W + x_idx).view(B, 1, -1)
        contrib = (ten_input * (weight * weight_part)).view(B, C, -1)
        output.view(B, C, -1).scatter_add_(2, idx.expand(-1, C, -1), contrib)
        accum.view(B, 1, -1).scatter_add_(2, idx, (weight * weight_part).view(B, 1, -1))

    scatter(wa, x0, y0)
    scatter(wb, x0, y1)
    scatter(wc, x1, y0)
    scatter(wd, x1, y1)

    output = output / torch.clamp(accum, min=eps)
    return output, accum

try:
    from simple_lama_inpainting import SimpleLama  # type: ignore
except ImportError:
    SimpleLama = None

_INPAINT_LOGGED = False
_WARP_LOGGED = False

# ============================================================
# PARTIE DEPTH ANYTHING V2
# ============================================================

def _get_depth_map_depth_anything(image_path, model_path=None):
    print(f"   Traitement profondeur pour : {os.path.basename(image_path)}")
    model_id = model_path or get_depth_model_id()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    
    depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False)
    depth_np = depth.squeeze().cpu().numpy()
    
    # Normalisation 0-1
    depth_min, depth_max = depth_np.min(), depth_np.max()
    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-6)
    
    # --- CORRECTIF 1 : SUPPRESSION DU BRUIT DANS LE CIEL ---
    # On force tout ce qui est très loin (valeur proche de 0) à devenir strictement 0.
    # Cela évite que le ciel soit considéré comme une surface bosselée à 10m de distance.
    SKY_THRESHOLD = 0.03  # Seuil de 3%
    depth_normalized[depth_normalized < SKY_THRESHOLD] = 0.0
    
    del model, processor, inputs, depth
    torch.cuda.empty_cache()
    
    return depth_normalized, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_depth_map(
    image_path,
    model_choice=None,
    model_id=None,
    unet_model_path=None,
):
    print(f"   Traitement profondeur pour : {os.path.basename(image_path)}")
    return predict_depth_map(
        image_path,
        model_choice=model_choice,
        model_id=model_id,
        unet_model_path=unet_model_path,
    )


# ============================================================
# PARTIE WARPING 3D (Adaptée FOV 90° Street View)
# ============================================================
def warp_image_3d(img_rgb, depth_norm, shift_z):
    global _WARP_LOGGED
    if not _WARP_LOGGED:
        print("[Interpolate] Warping: Softmax Splatting actif (occlusion softmax, pas de Z-buffer).")
        _WARP_LOGGED = True
    h, w = img_rgb.shape[:2]

    fov_degrees = 90
    fx = (w / 2) / math.tan(math.radians(fov_degrees / 2))
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    device = _get_device()

    depth_t = torch.from_numpy(depth_norm).to(device=device, dtype=torch.float32)
    img_t = (
        torch.from_numpy(img_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
        / 255.0
    )

    # Anchor sky far away, objects follow inverse depth
    z_metric = torch.zeros_like(depth_t)
    mask_sky = depth_t <= 0.0
    mask_obj = ~mask_sky
    z_metric[mask_sky] = 10000.0
    z_metric[mask_obj] = 1.0 / (depth_t[mask_obj] + 0.01)

    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x3d = (u - cx) * z_metric / fx
    y3d = (v - cy) * z_metric / fy
    z3d = z_metric

    z3d_new = z3d - shift_z
    valid_mask = z3d_new > 0.1

    u_new = (x3d * fx / z3d_new) + cx
    v_new = (y3d * fy / z3d_new) + cy

    flow = torch.zeros((1, 2, h, w), device=device, dtype=torch.float32)
    flow[:, 0] = u_new - u
    flow[:, 1] = v_new - v

    metric = -z3d_new.clone()
    metric[mask_sky] = -1.0
    metric[~valid_mask] = -1e8
    metric = metric.unsqueeze(0).unsqueeze(0)

    warped_img_t, weights = _softmax_splat(img_t, flow, metric)
    warped_depth_t, _ = _softmax_splat(z3d_new.unsqueeze(0).unsqueeze(0), flow, metric)

    mask = (weights.squeeze().detach().cpu().numpy() > 1e-6).astype(np.uint8) * 255
    warped_img = (
        warped_img_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255.0
    ).astype(np.uint8)

    warped_depth = warped_depth_t.squeeze().detach().cpu().numpy()
    warped_depth = np.nan_to_num(warped_depth, nan=1e4, posinf=1e4, neginf=1e4)

    return warped_img, warped_depth, mask
def consensus_merge(warp_a, depth_a, mask_a, warp_b, depth_b, mask_b, t):
    # 1. Masques de validité de base
    valid_a = mask_a > 0
    valid_b = mask_b > 0
    final_img = np.zeros_like(warp_a)
    
    # Zones exclusives
    mask_only_a = valid_a & (~valid_b)
    final_img[mask_only_a] = warp_a[mask_only_a]
    
    mask_only_b = valid_b & (~valid_a)
    final_img[mask_only_b] = warp_b[mask_only_b]
    
    # --- ZONE DE CHEVAUCHEMENT (LA OU TOUT SE JOUE) ---
    both = valid_a & valid_b
    if np.any(both):
        da = depth_a[both]
        db = depth_b[both]
        wa = warp_a[both].astype(np.float32)
        wb = warp_b[both].astype(np.float32)

        # --- NOUVEAUTÉ : DÉTECTION DE CHANGEMENT D'ÉTAT ---
        # On calcule la différence de couleur entre l'image warpée A et B
        # Si la différence est forte, c'est que l'objet a changé (portes ouvertes vs fermées)
        diff_color = np.mean(np.abs(wa - wb), axis=1) # Moyenne sur RGB
        
        # Seuil de tolérance (0-255). 
        # Si diff > 30, on considère que ce n'est pas le même objet visuel
        # (ex: Camion blanc vs Intérieur sombre)
        CHANGE_THRESHOLD = 30.0 
        mask_changed = diff_color > CHANGE_THRESHOLD

        # --- LOGIQUE HYBRIDE ---
        
        # 1. Poids par défaut basés sur le temps (Fondu simple)
        # C'est la base de sécurité : plus on avance (t augmente), plus on veut B.
        weights = np.full_like(da, t) 

        # 2. Logique d'Occlusion (Strict) - SEULEMENT SI PAS DE CHANGEMENT MAJEUR
        # On ne l'applique que sur les zones "stables" (mask_changed == False)
        # Si A est nettement devant B, on force A (poids=0).
        # Si B est nettement devant A, on force B (poids=1).
        
        occ_threshold = 0.5
        
        # Cas A devant B (ex: Poteau devant mur) -> On garde A (w=0)
        # MAIS on ignore cette règle si ça a changé (le camion)
        mask_a_wins = (da < (db - occ_threshold)) & (~mask_changed)
        weights[mask_a_wins] = 0.0
        
        # Cas B devant A -> On garde B (w=1)
        mask_b_wins = (db < (da - occ_threshold)) & (~mask_changed)
        weights[mask_b_wins] = 1.0

        # --- 3. ACCÉLÉRATION DU REMPLACEMENT POUR LES OBJETS CHANGEANTS ---
        # Si c'est une zone qui change (le camion), on veut basculer plus vite vers B
        # pour éviter de voir l'image fantôme de A trop longtemps.
        # On force la transition à être plus "tranchée".
        # Si t > 0.3, on force l'affichage de B sur les zones changeantes.
        mask_force_switch = mask_changed & (t > 0.3)
        weights[mask_force_switch] = np.clip(weights[mask_force_switch] * 2.0, 0, 1) # On booste B

        # Calcul final du pixel
        # On utilise weights[:, None] pour appliquer le même poids aux 3 canaux RGB
        w_expanded = weights[:, None]
        res = wa * (1.0 - w_expanded) + wb * w_expanded
        
        final_img[both] = res.astype(np.uint8)
        
    # Inpainting des trous
    mask_holes = (~(valid_a | valid_b)).astype(np.uint8) * 255
    if np.any(mask_holes):
        try:
            global _INPAINT_LOGGED
            if not _INPAINT_LOGGED:
                print("[Interpolate] Inpainting: LaMa (simple-lama-inpainting) utilisee pour combler les trous.")
                _INPAINT_LOGGED = True
            final_img = lama_inpaint(final_img, mask_holes)
        except Exception as exc:
            # Fallback to a lighter OpenCV method if LaMa is unavailable
            print(f"[Warn] LaMa inpainting failed ({exc}). Falling back to OpenCV inpaint.")
            final_img = cv2.inpaint(final_img, mask_holes, 3, cv2.INPAINT_NS)
        
    return final_img


# ============================================================
# PIPELINE (Adapté pour step=1m)
# ============================================================

def process_interpolation(
    img_a_path,
    img_b_path,
    output_frames_folder,
    num_frames=30,
    start_index=0,
    skip_first_frame=False,
    distance_hint_m=None,
    depth_model_choice="depth-anything",
    depth_model_id=None,
    unet_model_path=None,
):
    # Trace les moteurs actifs pour cette interpolation
    inpaint_engine = "LaMa" if SimpleLama is not None else "OpenCV_NS"
    print("[Interpolate] Engines: warp=SoftmaxSplatting | inpaint=%s | scale=distance_hint" % inpaint_engine)

    model_label = depth_model_choice or "depth-anything"
    print(f">>> [Interpolate] 1. Calcul des Depth Maps ({model_label})...")
    depth_a, img_a_cv = get_depth_map(
        img_a_path,
        model_choice=depth_model_choice,
        model_id=depth_model_id,
        unet_model_path=unet_model_path,
    )
    depth_b, img_b_cv = get_depth_map(
        img_b_path,
        model_choice=depth_model_choice,
        model_id=depth_model_id,
        unet_model_path=unet_model_path,
    )
    
    # 2. Scale basé sur la distance réelle
    print(">>> [Interpolate] 2. Calcul du scale...")
    
    # --- CONFIGURATION DU SCALE ---
    # On garde la valeur FORTE (0.28) qui donne le bon alignement au début.
    # C'est la "Vitesse Initiale".
    scale = 0.35 # Fallback
    
    if distance_hint_m is not None and distance_hint_m > 0:
        # On reste sur le ratio agressif que tu as validé pour le début
        REAL_TO_VIRTUAL_RATIO = 0.50
        scale = distance_hint_m * REAL_TO_VIRTUAL_RATIO
        print(f"    [Info] Distance réelle: {distance_hint_m:.2f}m -> Scale virtuel de base: {scale:.4f}")
    else:
        scale = 0.35
        print("    [Warn] Pas de distance fournie. Scale par défaut appliqué: 0.35")

    # 3. Génération avec FREINAGE DYNAMIQUE
    print(f">>> [Interpolate] 3. Génération de {num_frames} frames...")
    
    frames_written = 0
    current_index = start_index

    # --- PARAMETRE DE FREINAGE ---
    # 0.0 = Mouvement linéaire (Vitesse constante)
    # 0.3 = On parcourt 30% de distance en moins à la fin (Freinage doux)
    # C'est ce qui permet d'avoir un Scale 0.28 au début mais équivalent à 0.20 à la fin.
    BRAKING_FORCE = 0.36
    
    # Facteur final de distance (ex: 0.7)
    MAX_DIST_FACTOR = 1.0 - BRAKING_FORCE

    for i in range(num_frames):
        if skip_first_frame and i == 0:
            continue

        # t_blend : Progression temporelle linéaire (0 -> 1)
        # Sert uniquement à gérer le fondu (transparence) entre les images
        t_blend = i / (num_frames - 1)
        
        # t_geo : Progression géométrique (Courbe quadratique)
        # Formule : t * (1 - k*t)
        # À t=0, la vitesse est 100% (respecte le scale 0.28)
        # À t=1, la position est 70% (respecte l'alignement de fin)
        t_geo = t_blend * (1.0 - BRAKING_FORCE * t_blend)
        
        # Calcul des shifts basé sur ce temps "freiné"
        
        # Camera A avance de 0 à 0.7*scale
        shift_a = scale * t_geo
        
        # Camera B est située virtuellement à 0.7*scale. On calcule le recul nécessaire.
        # shift_b part de -0.7*scale (à t=0) pour arriver à 0 (à t=1)
        shift_b = scale * (t_geo - MAX_DIST_FACTOR)
        
        wa, da, ma = warp_image_3d(img_a_cv, depth_a, shift_a)
        wb, db, mb = warp_image_3d(img_b_cv, depth_b, shift_b)
        
        # On passe t_blend (linéaire) au consensus pour que le fondu reste fluide
        final = consensus_merge(wa, da, ma, wb, db, mb, t_blend)
        
        path = os.path.join(output_frames_folder, f"frame_{current_index:03d}.jpg")
        cv2.imwrite(path, final)
        frames_written += 1
        current_index += 1
        
        if i % 10 == 0:
            print(f"    ... Frame {frames_written}/{num_frames if not skip_first_frame else num_frames-1}")

    torch.cuda.empty_cache()
    return frames_written
