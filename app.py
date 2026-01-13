import glob
import json
import os
import shutil
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

import cv2
import numpy as np
from PIL import Image

import depth_metrics
import generate_frames
import interpolate
from config_loader import get_api_key, get_depth_model_id, get_unet_model_path, load_config
from depth_models import render_depth_colormap


APP_ROOT = os.path.abspath(os.path.dirname(__file__))
BASE_OUTPUT = "street_view_project_output"
os.makedirs(BASE_OUTPUT, exist_ok=True)
GT_SAMPLE_ROOT = os.path.join(APP_ROOT, "data_train_test")
GT_PHOTO_DIR = os.path.join(GT_SAMPLE_ROOT, "photos")
GT_DEPTH_DIR = os.path.join(GT_SAMPLE_ROOT, "depth")
GT_MAX_SAMPLES = 100

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
DEFAULT_STREETVIEW_COUNT = int(os.environ.get("STREETVIEW_PHOTOS", "4"))
CONFIG = load_config()
DEFAULT_API_KEY = get_api_key(CONFIG)
DEPTH_MODEL_ID = get_depth_model_id(CONFIG)
DEFAULT_UNET_MODEL_PATH = get_unet_model_path(CONFIG)
VIDEO_TARGET_SPEED_M_S = float(os.environ.get("VIDEO_TARGET_SPEED_M_S", "8.0"))
VIDEO_DEFAULT_FPS = float(os.environ.get("VIDEO_DEFAULT_FPS", "24.0"))
VIDEO_MIN_FPS = 8.0
VIDEO_MAX_FPS = 60.0
# Toggle en dur: True = distance_hint GPS prioritaire (pas de RoMa/SIFT)
USE_ROMA_FOR_SCALE = True
os.environ["USE_ROMA_FOR_SCALE"] = "1" if USE_ROMA_FOR_SCALE else "0"

# Memoire simple pour l'etat des executions en cours
RUNS = {}


def _slugify(value):
    safe = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def list_local_unet_models():
    pattern = os.path.join(APP_ROOT, "*.pth")
    return sorted(os.path.basename(path) for path in glob.glob(pattern))


def list_available_model_names(depth_model_id, unet_model_path):
    names = []
    seen = set()
    for name in list_local_unet_models():
        if name not in seen:
            names.append(name)
            seen.add(name)
    if unet_model_path and os.path.isfile(unet_model_path):
        name = unet_model_path
        if os.path.basename(unet_model_path) in seen:
            name = os.path.basename(unet_model_path)
        if name not in seen:
            names.append(name)
            seen.add(name)
    if depth_model_id and depth_model_id not in seen:
        names.append(depth_model_id)
        seen.add(depth_model_id)
    return names


def default_model_name(depth_model_id, unet_model_path):
    local_unets = list_local_unet_models()
    if local_unets:
        return local_unets[0]
    if unet_model_path and os.path.isfile(unet_model_path):
        return os.path.basename(unet_model_path)
    return depth_model_id


def pick_alternate_model(names, primary):
    for name in names:
        if name != primary:
            return name
    return primary


def resolve_depth_model_selection(model_name, depth_model_id, unet_model_path):
    name = (model_name or "").strip()
    if not name:
        name = default_model_name(depth_model_id, unet_model_path)
    is_unet = name.lower().endswith(".pth")
    if is_unet:
        resolved = name
        if not os.path.isabs(resolved):
            resolved = os.path.join(APP_ROOT, resolved)
        if not os.path.isfile(resolved):
            fallback = None
            if unet_model_path and os.path.basename(unet_model_path) == name:
                fallback = unet_model_path
            if fallback and os.path.isfile(fallback):
                resolved = fallback
            else:
                raise FileNotFoundError(f"Modele UNet introuvable: {name}")
        return "unet", depth_model_id, resolved, name
    return "depth-anything", name or depth_model_id, unet_model_path, name


def build_depth_model_key(model_choice, model_name):
    if model_choice == "unet":
        stem = os.path.splitext(os.path.basename(model_name or ""))[0] or "unet"
        slug = _slugify(stem)
        return f"unet-{slug}" if slug else "unet"
    slug = _slugify(model_name or "depth-anything")
    return f"depth-anything-{slug}" if slug else "depth-anything"


def build_depth_model_label(model_choice, model_name, depth_model_id, unet_model_path):
    if model_choice == "unet":
        name = os.path.basename(unet_model_path or model_name or "UNet")
        return f"UNet ({name})"
    name = model_name or depth_model_id
    return f"Depth Anything ({name})"


def label_from_model_key(model_key):
    if model_key.startswith("unet-"):
        label = model_key.split("-", 1)[1]
        return f"UNet ({label})" if label else "UNet"
    if model_key.startswith("depth-anything-"):
        label = model_key.split("-", 1)[1]
        return f"Depth Anything ({label})" if label else "Depth Anything"
    if model_key.startswith("unet"):
        return "UNet"
    if model_key.startswith("depth-anything"):
        return "Depth Anything"
    return model_key


def build_run_id():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def metrics_output_dir(run_id, direction, model_key_a, model_key_b):
    pair = f"{model_key_a}__vs__{model_key_b}"
    return ensure_dir(os.path.join(BASE_OUTPUT, run_id, "model_metrics", direction, pair))


def ground_truth_output_dir(run_id, model_key_a, model_key_b):
    pair = f"{model_key_a}__vs__{model_key_b}"
    return ensure_dir(os.path.join(BASE_OUTPUT, run_id, "model_metrics", "ground_truth", pair))


def load_ground_truth_depth(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth_img = Image.open(path)
            depth = np.array(depth_img, dtype=np.float32)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Ground truth depth not found: {path}") from exc
    if depth.ndim == 3 and depth.shape[0] == 1 and depth.shape[2] != 1:
        depth = np.transpose(depth, (1, 2, 0))
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros_like(depth, dtype=np.float32), 0.0
    depth_valid = depth[valid]
    depth_min = float(depth_valid.min())
    depth_max = float(depth_valid.max())
    if depth_max - depth_min < 1e-6:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_norm = np.where(valid, depth_norm, 0.0).astype(np.float32)
    valid_ratio = float(valid.sum() / valid.size)
    return depth_norm, valid_ratio


def list_ground_truth_pairs(max_items: int = GT_MAX_SAMPLES):
    if not os.path.isdir(GT_PHOTO_DIR) or not os.path.isdir(GT_DEPTH_DIR):
        return []
    photo_files = glob.glob(os.path.join(GT_PHOTO_DIR, "*.npy"))
    depth_files = glob.glob(os.path.join(GT_DEPTH_DIR, "*.npy"))
    if not photo_files or not depth_files:
        return []
    photo_map = {os.path.splitext(os.path.basename(p))[0]: p for p in photo_files}
    depth_map = {os.path.splitext(os.path.basename(p))[0]: p for p in depth_files}
    shared = [stem for stem in photo_map.keys() if stem in depth_map]
    if not shared:
        return []

    def sort_key(stem):
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    shared.sort(key=sort_key)
    pairs = [
        {
            "stem": stem,
            "photo": photo_map[stem],
            "depth": depth_map[stem],
        }
        for stem in shared
    ]
    if max_items:
        return pairs[:max_items]
    return pairs


def _save_ground_truth_input(src_path: str, dest_path: str) -> str:
    ext = os.path.splitext(src_path)[1].lower()
    if ext == ".npy":
        arr = np.load(src_path)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            max_val = float(np.nanmax(arr)) if arr.size else 0.0
            if max_val <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(dest_path)
        return dest_path
    shutil.copyfile(src_path, dest_path)
    return dest_path


def list_available_runs():
    """Retourne les dossiers disponibles dans la sortie, triés par date."""
    if not os.path.isdir(BASE_OUTPUT):
        return []
    entries = []
    for name in os.listdir(BASE_OUTPUT):
        full = os.path.join(BASE_OUTPUT, name)
        if os.path.isdir(full):
            entries.append((name, os.path.getmtime(full)))
    entries.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in entries]


def clean_folder(path):
    if not os.path.isdir(path):
        return
    for fname in os.listdir(path):
        full = os.path.join(path, fname)
        if os.path.isfile(full):
            os.remove(full)


def list_sources_from_disk(run_id, direction):
    folder = os.path.join(BASE_OUTPUT, run_id, direction, "sources")
    if not os.path.isdir(folder):
        return []
    patterns = [os.path.join(folder, "*.jpg"), os.path.join(folder, "*.jpeg"), os.path.join(folder, "*.png")]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(found)


def get_sources_for_run(run_id, direction):
    run = RUNS.get(run_id)
    if run:
        return run.get("paths", {}).get(direction, {}).get("sources", []) or []
    return list_sources_from_disk(run_id, direction)


def _cleanup_outputs_after_source_change(run_id, direction):
    frames_dir = os.path.join(BASE_OUTPUT, run_id, "frames", direction)
    clean_folder(frames_dir)
    previews_dir = os.path.join(BASE_OUTPUT, run_id, "depth_previews", direction)
    if os.path.isdir(previews_dir):
        shutil.rmtree(previews_dir, ignore_errors=True)
    video_path = os.path.join(BASE_OUTPUT, run_id, "videos", f"{direction}.mp4")
    if os.path.isfile(video_path):
        os.remove(video_path)
    run = RUNS.get(run_id)
    if run:
        (run.get("interpolation", {}) or {}).pop(direction, None)


def list_depth_previews(run_id):
    previews = {"forward": [], "backward": []}
    base_dir = os.path.join(BASE_OUTPUT, run_id, "depth_previews")
    if not os.path.isdir(base_dir):
        return previews
    for direction in ("forward", "backward"):
        dir_path = os.path.join(base_dir, direction)
        if not os.path.isdir(dir_path):
            continue
        for model_key in sorted(os.listdir(dir_path)):
            model_dir = os.path.join(dir_path, model_key)
            if not os.path.isdir(model_dir):
                continue
            images = sorted(glob.glob(os.path.join(model_dir, "*.jpg")))
            if not images:
                continue
            label = label_from_model_key(model_key)
            meta_path = os.path.join(model_dir, "model.txt")
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as meta:
                    label = meta.read().strip() or label
            previews[direction].append(
                {
                    "label": label,
                    "images": [path_to_url(p) for p in images],
                }
            )
    return previews


def summarize_metrics(per_image):
    keys = ("mae", "rmse", "pearson_r", "edge_f1", "valid_ratio")
    summary = {}
    for key in keys:
        values = [item.get(key) for item in per_image if item.get(key) is not None]
        summary[key] = (sum(values) / len(values)) if values else None
    summary["image_count"] = len(per_image)
    summary["valid_image_count"] = sum(
        1 for item in per_image if (item.get("valid_ratio") or 0.0) > 0.0
    )
    return summary


def format_metric(value, fmt):
    if value is None:
        return "n/a"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def write_metrics_txt(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def format_comparison_metrics_txt(payload):
    summary = payload.get("summary", {})
    lines = [
        "Model comparison metrics",
        f"created_at: {payload.get('created_at')}",
        f"direction: {payload.get('direction')}",
        f"model_a: {payload.get('model_a', {}).get('label')}",
        f"model_b: {payload.get('model_b', {}).get('label')}",
        "summary:",
        f"  mae: {format_metric(summary.get('mae'), '.6f')}",
        f"  rmse: {format_metric(summary.get('rmse'), '.6f')}",
        f"  pearson_r: {format_metric(summary.get('pearson_r'), '.6f')}",
        f"  edge_f1: {format_metric(summary.get('edge_f1'), '.6f')}",
        f"  valid_ratio: {format_metric(summary.get('valid_ratio'), '.6f')}",
        f"  image_count: {summary.get('image_count')}",
        f"  valid_image_count: {summary.get('valid_image_count')}",
        f"  min_valid_depth: {summary.get('min_valid_depth')}",
        f"  duration_s: {format_metric(summary.get('duration_s'), '.3f')}",
        "per_image:",
    ]
    for item in payload.get("per_image", []):
        lines.append(
            "  - {image}: mae={mae} rmse={rmse} pearson_r={pearson_r} edge_f1={edge_f1} valid_ratio={valid_ratio}".format(
                image=item.get("image"),
                mae=format_metric(item.get("mae"), ".6f"),
                rmse=format_metric(item.get("rmse"), ".6f"),
                pearson_r=format_metric(item.get("pearson_r"), ".6f"),
                edge_f1=format_metric(item.get("edge_f1"), ".6f"),
                valid_ratio=format_metric(item.get("valid_ratio"), ".6f"),
            )
        )
    return lines


def format_groundtruth_metrics_txt(payload):
    model_a = payload.get("model_a", {})
    model_b = payload.get("model_b", {})
    gt = payload.get("ground_truth", {})
    lines = [
        "Ground truth metrics (batch)",
        f"created_at: {payload.get('created_at')}",
        f"sample_count: {payload.get('sample', {}).get('count')}",
        f"photos_dir: {payload.get('sample', {}).get('photos_dir')}",
        f"depth_dir: {payload.get('sample', {}).get('depth_dir')}",
        f"model_a: {model_a.get('label')}",
        f"model_b: {model_b.get('label')}",
        f"ground_truth_min_valid_depth: {gt.get('min_valid_depth')}",
        f"ground_truth_valid_ratio_avg: {format_metric(gt.get('valid_ratio'), '.6f')}",
        "model_a_summary:",
        f"  mae: {format_metric(model_a.get('metrics', {}).get('mae'), '.6f')}",
        f"  rmse: {format_metric(model_a.get('metrics', {}).get('rmse'), '.6f')}",
        f"  pearson_r: {format_metric(model_a.get('metrics', {}).get('pearson_r'), '.6f')}",
        f"  edge_f1: {format_metric(model_a.get('metrics', {}).get('edge_f1'), '.6f')}",
        f"  valid_ratio: {format_metric(model_a.get('metrics', {}).get('valid_ratio'), '.6f')}",
        "model_b_summary:",
        f"  mae: {format_metric(model_b.get('metrics', {}).get('mae'), '.6f')}",
        f"  rmse: {format_metric(model_b.get('metrics', {}).get('rmse'), '.6f')}",
        f"  pearson_r: {format_metric(model_b.get('metrics', {}).get('pearson_r'), '.6f')}",
        f"  edge_f1: {format_metric(model_b.get('metrics', {}).get('edge_f1'), '.6f')}",
        f"  valid_ratio: {format_metric(model_b.get('metrics', {}).get('valid_ratio'), '.6f')}",
        "per_image:",
    ]
    for item in payload.get("per_image", []):
        lines.append(
            "  - {image}: a_mae={a_mae} a_rmse={a_rmse} a_r={a_r} a_edge={a_edge} | b_mae={b_mae} b_rmse={b_rmse} b_r={b_r} b_edge={b_edge} gt_valid={gt_valid}".format(
                image=item.get("image"),
                a_mae=format_metric(item.get("model_a", {}).get("mae"), ".6f"),
                a_rmse=format_metric(item.get("model_a", {}).get("rmse"), ".6f"),
                a_r=format_metric(item.get("model_a", {}).get("pearson_r"), ".6f"),
                a_edge=format_metric(item.get("model_a", {}).get("edge_f1"), ".6f"),
                b_mae=format_metric(item.get("model_b", {}).get("mae"), ".6f"),
                b_rmse=format_metric(item.get("model_b", {}).get("rmse"), ".6f"),
                b_r=format_metric(item.get("model_b", {}).get("pearson_r"), ".6f"),
                b_edge=format_metric(item.get("model_b", {}).get("edge_f1"), ".6f"),
                gt_valid=format_metric(item.get("gt_valid_ratio"), ".6f"),
            )
        )
    return lines


def ensure_metrics_txt(metrics_json_path, payload, kind):
    metrics_txt_path = os.path.join(os.path.dirname(metrics_json_path), "metrics.txt")
    if os.path.isfile(metrics_txt_path):
        return
    try:
        if kind == "ground_truth":
            lines = format_groundtruth_metrics_txt(payload)
        else:
            lines = format_comparison_metrics_txt(payload)
        write_metrics_txt(metrics_txt_path, lines)
    except OSError:
        return


def list_model_evaluations(run_id):
    base_dir = os.path.join(BASE_OUTPUT, run_id, "model_metrics")
    if not os.path.isdir(base_dir):
        return []
    paths = glob.glob(os.path.join(base_dir, "**", "metrics.json"), recursive=True)
    evaluations = []
    for path in paths:
        if os.path.sep + "ground_truth" + os.path.sep in path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        ensure_metrics_txt(path, data, "comparison")
        evaluations.append(data)
    evaluations.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return evaluations


def list_ground_truth_evaluations(run_id):
    base_dir = os.path.join(BASE_OUTPUT, run_id, "model_metrics", "ground_truth")
    if not os.path.isdir(base_dir):
        return []
    paths = glob.glob(os.path.join(base_dir, "**", "metrics.json"), recursive=True)
    evaluations = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        ensure_metrics_txt(path, data, "ground_truth")
        images = data.get("images", {})
        image_urls = {}
        for key, rel_path in images.items():
            if not rel_path:
                image_urls[key] = None
                continue
            full_path = os.path.join(BASE_OUTPUT, rel_path)
            image_urls[key] = path_to_url(full_path)
        data["image_urls"] = image_urls
        evaluations.append(data)
    evaluations.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return evaluations


def list_frames(folder):
    if not os.path.isdir(folder):
        return []
    frames = sorted(glob.glob(os.path.join(folder, "frame_*.jpg")))
    return [url_for("serve_output", filename=os.path.relpath(frame, BASE_OUTPUT).replace("\\", "/")) for frame in frames]


def list_frame_paths(folder):
    if not os.path.isdir(folder):
        return []
    frames = glob.glob(os.path.join(folder, "frame_*.jpg"))
    def extract_idx(path):
        try:
            return int(os.path.splitext(os.path.basename(path))[0].split("_")[1])
        except (ValueError, IndexError):
            return 0

    return sorted(frames, key=extract_idx)


def path_to_url(path):
    return url_for("serve_output", filename=os.path.relpath(path, BASE_OUTPUT).replace("\\", "/"))


def compute_video_fps(total_distance_m, frames_per_official, segments, target_speed_m_s=VIDEO_TARGET_SPEED_M_S):
    """
    Calcule une cadence qui respecte la distance reelle entre panoramas et
    le nombre de frames generes par intervalle officiel.
    """
    if total_distance_m and frames_per_official and segments > 0:
        meters_per_segment = total_distance_m / segments
        meters_per_frame = meters_per_segment / frames_per_official
        fps = target_speed_m_s / max(meters_per_frame, 1e-3)
        return max(VIDEO_MIN_FPS, min(VIDEO_MAX_FPS, fps))
    return VIDEO_DEFAULT_FPS


def log_console(message):
    """Petit helper pour tracer dans la console avec flush explicite."""
    print(message, flush=True)


def print_progress(label, current, total, width=24):
    """Affiche une barre ASCII simple pour suivre l'avancement."""
    if total <= 0:
        log_console(f"{label} [no progress available]")
        return
    ratio = min(max(float(current) / float(total), 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    log_console(f"{label} [{bar}] {current}/{total} ({ratio*100:.1f}%)")


def write_video(frame_paths, output_path, fps):
    if not frame_paths:
        raise ValueError("Aucune frame disponible pour la video.")

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError("Impossible de lire la premiere frame.")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    if not os.path.isfile(output_path):
        raise RuntimeError("Ecriture video echouee.")


def generate_depth_previews(run_id, direction, model_name, depth_model_id, unet_model_path):
    sources = get_sources_for_run(run_id, direction)
    if len(sources) < 1:
        raise ValueError("Aucune image source disponible pour ce sens.")
    model_choice, depth_model_id, unet_model_path, resolved_name = resolve_depth_model_selection(
        model_name, depth_model_id, unet_model_path
    )
    model_key = build_depth_model_key(model_choice, resolved_name)
    output_dir = ensure_dir(os.path.join(BASE_OUTPUT, run_id, "depth_previews", direction, model_key))
    clean_folder(output_dir)
    model_label = build_depth_model_label(model_choice, resolved_name, depth_model_id, unet_model_path)
    meta_path = os.path.join(output_dir, "model.txt")
    with open(meta_path, "w", encoding="utf-8") as meta:
        meta.write(model_label)
    for idx, source_path in enumerate(sources):
        depth_map, _ = interpolate.get_depth_map(
            source_path,
            model_choice=model_choice,
            model_id=depth_model_id,
            unet_model_path=unet_model_path,
        )
        colored = render_depth_colormap(depth_map)
        out_path = os.path.join(output_dir, f"depth_{idx:02d}.jpg")
        cv2.imwrite(out_path, colored)
    return output_dir


@app.route("/", methods=["GET", "POST"])
def index():
    default_key = DEFAULT_API_KEY
    available_runs = list_available_runs()

    if request.method == "POST":
        address = request.form.get("address", "").strip()
        api_key = request.form.get("api_key", "").strip() or default_key
        num_sources = max(2, int(request.form.get("num_sources", DEFAULT_STREETVIEW_COUNT)))

        if not address or not api_key:
            flash("Adresse et cle API sont obligatoires.")
            return redirect(url_for("index"))

        run_id = build_run_id()
        run_root = ensure_dir(os.path.join(BASE_OUTPUT, run_id))
        log_console(f"[Run] Nouvelle session {run_id} pour '{address}' avec {num_sources} panoramas par sens.")

        dirs = {
            "forward": {"sources": ensure_dir(os.path.join(run_root, "forward", "sources"))},
            "backward": {"sources": ensure_dir(os.path.join(run_root, "backward", "sources"))},
        }

        try:
            forward_sources, forward_meta = generate_frames.fetch_source_images(
                address=address,
                api_key=api_key,
                output_folder=dirs["forward"]["sources"],
                inverser_sens=False,
                num_sources=num_sources,
                return_meta=True,
            )
            backward_sources, backward_meta = generate_frames.fetch_source_images(
                address=address,
                api_key=api_key,
                output_folder=dirs["backward"]["sources"],
                inverser_sens=True,
                num_sources=num_sources,
                return_meta=True,
            )
            log_console(f"[Run] Images forward : {len(forward_sources)} | Images backward : {len(backward_sources)}")
        except Exception as exc:
            flash(f"Erreur pendant la generation des images : {exc}")
            return redirect(url_for("index"))

        if len(forward_sources) < 2 or len(backward_sources) < 2:
            flash("Impossible de recuperer suffisamment d'images officielles.")
            return redirect(url_for("index"))

        RUNS[run_id] = {
            "address": address,
            "api_key": api_key,
            "num_sources": num_sources,
            "paths": {
                "forward": {"sources": forward_sources, "meta": forward_meta},
                "backward": {"sources": backward_sources, "meta": backward_meta},
            },
            "video_options": {"speed_m_s": VIDEO_TARGET_SPEED_M_S},
        }

        return redirect(url_for("index", run_id=run_id))

    run_id = request.args.get("run_id")
    run = RUNS.get(run_id) if run_id else None

    forward_items = backward_items = None
    frames_forward = frames_backward = []
    video_forward = video_backward = None
    address = ""
    num_sources = DEFAULT_STREETVIEW_COUNT
    video_speed_m_s = VIDEO_TARGET_SPEED_M_S
    depth_model_names = list_available_model_names(DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH)
    depth_model_name = default_model_name(DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH)
    depth_previews = {"forward": [], "backward": []}
    model_compare_a = depth_model_name
    model_compare_b = pick_alternate_model(depth_model_names, model_compare_a)
    model_evaluations = []
    ground_truth_evaluations = []
    gt_sample_ready = bool(list_ground_truth_pairs(1))

    if run_id:
        frames_forward = list_frames(os.path.join(BASE_OUTPUT, run_id, "frames", "forward"))
        frames_backward = list_frames(os.path.join(BASE_OUTPUT, run_id, "frames", "backward"))
        depth_previews = list_depth_previews(run_id)
        model_evaluations = list_model_evaluations(run_id)
        ground_truth_evaluations = list_ground_truth_evaluations(run_id)

        if not run:
            forward_disk = list_sources_from_disk(run_id, "forward")
            backward_disk = list_sources_from_disk(run_id, "backward")
            if forward_disk or backward_disk:
                # Reconstruit un run minimal pour permettre l'affichage dans la galerie.
                run = RUNS[run_id] = {
                    "address": "",
                    "api_key": "",
                    "num_sources": max(len(forward_disk), len(backward_disk)),
                    "paths": {
                        "forward": {"sources": forward_disk, "meta": []},
                        "backward": {"sources": backward_disk, "meta": []},
                    },
                    "video_options": {"speed_m_s": VIDEO_TARGET_SPEED_M_S},
                }

    if run:
        address = run.get("address", "")
        num_sources = run.get("num_sources", DEFAULT_STREETVIEW_COUNT)
        depth_settings = run.get("depth_settings", {})
        depth_model_name = depth_settings.get("model_name", depth_model_name)
        model_compare_a = depth_model_name
        if depth_model_name and depth_model_name not in depth_model_names:
            depth_model_names.append(depth_model_name)
        model_compare_b = pick_alternate_model(depth_model_names, model_compare_a)
        forward = run.get("paths", {}).get("forward", {}).get("sources", []) or []
        backward = run.get("paths", {}).get("backward", {}).get("sources", []) or []
        if forward:
            forward_items = [
                {"url": path_to_url(path), "index": idx}
                for idx, path in enumerate(forward)
            ]
        if backward:
            backward_items = [
                {"url": path_to_url(path), "index": idx}
                for idx, path in enumerate(backward)
            ]
        video_forward_path = os.path.join(BASE_OUTPUT, run_id, "videos", "forward.mp4")
        video_backward_path = os.path.join(BASE_OUTPUT, run_id, "videos", "backward.mp4")
        if os.path.isfile(video_forward_path):
            video_forward = path_to_url(video_forward_path)
        if os.path.isfile(video_backward_path):
            video_backward = path_to_url(video_backward_path)
        video_speed_m_s = run.get("video_options", {}).get("speed_m_s", VIDEO_TARGET_SPEED_M_S)

    return render_template(
        "run.html",
        run_id=run_id,
        address=address,
        forward=forward_items,
        backward=backward_items,
        frames_forward=frames_forward,
        frames_backward=frames_backward,
        video_forward=video_forward,
        video_backward=video_backward,
        video_speed_m_s=video_speed_m_s,
        default_key=default_key,
        num_sources=num_sources,
        available_runs=available_runs,
        depth_model_names=depth_model_names,
        depth_model_name=depth_model_name,
        depth_previews=depth_previews,
        model_compare_a=model_compare_a,
        model_compare_b=model_compare_b,
        model_evaluations=model_evaluations,
        ground_truth_evaluations=ground_truth_evaluations,
        gt_sample_ready=gt_sample_ready,
        model_eval_min_valid_depth=depth_metrics.DEFAULT_MIN_VALID_DEPTH,
    )


@app.route("/delete_source", methods=["POST"])
def delete_source():
    run_id = request.form.get("run_id")
    direction = request.form.get("direction", "forward")
    index_raw = request.form.get("index", "")
    if not run_id:
        flash("Session manquante pour la suppression.")
        return redirect(url_for("index"))
    if direction not in ("forward", "backward"):
        flash("Sens inconnu pour la suppression.")
        return redirect(url_for("index", run_id=run_id))
    try:
        index = int(index_raw)
    except (TypeError, ValueError):
        flash("Index invalide pour la suppression.")
        return redirect(url_for("index", run_id=run_id))

    sources = get_sources_for_run(run_id, direction)
    if not sources:
        flash("Aucune image source a supprimer.")
        return redirect(url_for("index", run_id=run_id))
    if len(sources) <= 2:
        flash("Suppression refusee : il faut au moins 2 images pour interpoler.")
        return redirect(url_for("index", run_id=run_id))
    if index < 0 or index >= len(sources):
        flash("Index hors limite pour la suppression.")
        return redirect(url_for("index", run_id=run_id))

    target_path = sources[index]
    sources_dir = os.path.abspath(os.path.join(BASE_OUTPUT, run_id, direction, "sources"))
    abs_target = os.path.abspath(target_path)
    if not abs_target.startswith(sources_dir):
        flash("Chemin source invalide.")
        return redirect(url_for("index", run_id=run_id))
    try:
        if os.path.isfile(abs_target):
            os.remove(abs_target)
    except OSError as exc:
        flash(f"Erreur lors de la suppression : {exc}")
        return redirect(url_for("index", run_id=run_id))

    run = RUNS.get(run_id)
    if run:
        run_sources = run.get("paths", {}).get(direction, {}).get("sources", [])
        if 0 <= index < len(run_sources):
            run_sources.pop(index)
        meta = run.get("paths", {}).get(direction, {}).get("meta", [])
        if meta and 0 <= index < len(meta):
            meta.pop(index)
        if meta and len(meta) == len(run_sources):
            for idx, item in enumerate(meta):
                if idx == 0:
                    item["distance_from_prev_m"] = 0.0
                    continue
                prev = meta[idx - 1]
                try:
                    item["distance_from_prev_m"] = float(
                        generate_frames.get_real_distance(
                            prev.get("lat"),
                            prev.get("lng"),
                            item.get("lat"),
                            item.get("lng"),
                        )
                    )
                except Exception:
                    item["distance_from_prev_m"] = item.get("distance_from_prev_m", 0.0)
        forward_len = len(run.get("paths", {}).get("forward", {}).get("sources", []) or [])
        backward_len = len(run.get("paths", {}).get("backward", {}).get("sources", []) or [])
        run["num_sources"] = max(forward_len, backward_len, 2)

    _cleanup_outputs_after_source_change(run_id, direction)
    flash("Image source supprimee.")
    return redirect(url_for("index", run_id=run_id))


@app.route("/interpolate", methods=["POST"])
def run_interpolation():
    run_id = request.form.get("run_id")
    run = RUNS.get(run_id)
    if not run:
        flash("Session introuvable, veuillez relancer une generation.")
        return redirect(url_for("index"))

    direction = request.form.get("direction", "forward")
    num_frames = max(2, int(request.form.get("frames_per_official", 30)))
    model_name = (request.form.get("model_name") or "").strip()
    try:
        depth_model_choice, depth_model_id, unet_model_path, resolved_name = resolve_depth_model_selection(
            model_name, DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH
        )
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index", run_id=run_id))

    if direction not in ("forward", "backward"):
        flash("Sens inconnu.")
        return redirect(url_for("index", run_id=run_id))
    sources = run["paths"][direction]["sources"]
    meta = run["paths"][direction].get("meta") or []
    distances = [item.get("distance_from_prev_m", None) for item in meta]
    if len(sources) < 2:
        flash("Pas assez d'images officielles pour interpoler.")
        return redirect(url_for("index", run_id=run_id))

    segments = len(sources) - 1
    frames_dir = ensure_dir(os.path.join(BASE_OUTPUT, run_id, "frames", direction))
    clean_folder(frames_dir)
    log_console(f"[Interpolate] Sens={direction} segments={segments} frames/segment={num_frames}")
    # Rappel du mode fixé en dur
    log_console(f"[Interpolate] Mode scale global: distance_hint (USE_ROMA_FOR_SCALE={int(USE_ROMA_FOR_SCALE)})")

    try:
        frame_index = 0

        # On commence toujours par la photo officielle source_00 pour eviter un premier frame dechiré.
        first_official_dest = os.path.join(frames_dir, f"frame_{frame_index:03d}.jpg")
        shutil.copyfile(sources[0], first_official_dest)
        frame_index += 1
        log_console(f"[Interpolate] Frame officielle initiale copiee -> {first_official_dest}")

        for idx in range(segments):
            distance_hint = distances[idx + 1] if distances and idx + 1 < len(distances) else None
            mode = "distance_hint" if distance_hint else "fallback_RoMa"

            label = f"[Interpolate] Segment {idx+1}/{segments} mode={mode}"
            print_progress(label, idx + 1, segments)

            written = interpolate.process_interpolation(
                img_a_path=sources[idx],
                img_b_path=sources[idx + 1],
                output_frames_folder=frames_dir,
                num_frames=num_frames,
                start_index=frame_index,
                skip_first_frame=True,
                distance_hint_m=distance_hint,
                depth_model_choice=depth_model_choice,
                depth_model_id=depth_model_id,
                unet_model_path=unet_model_path,
            )
            frame_index += written
            log_console(f"[Interpolate] Segment {idx+1} termine : {written} frames generees (total {frame_index}).")

            # Ajoute la photo officielle suivante pour reancrer la sequence
            # et eviter un frame final trop degrade par le warp.
            if idx + 1 < len(sources):
                official_path = os.path.join(frames_dir, f"frame_{frame_index:03d}.jpg")
                shutil.copyfile(sources[idx + 1], official_path)
                frame_index += 1
                log_console(f"[Interpolate] Frame officielle ajoutee -> {official_path}")

        total_distance = sum(float(d or 0.0) for d in distances) if distances else 0.0
        run.setdefault("interpolation", {})[direction] = {
            "frames_per_official": num_frames,
            "frames_generated": frame_index,
            "total_distance_m": total_distance,
        }
    except Exception as exc:
        flash(f"Erreur pendant l'interpolation : {exc}")
        return redirect(url_for("index", run_id=run_id))

    flash(f"{frame_index} frames generees pour le sens {direction} ({segments} segment(s)).")
    run.setdefault("depth_settings", {}).update(
        {
            "model_choice": depth_model_choice,
            "model_id": depth_model_id,
            "unet_model_path": unet_model_path,
            "model_name": resolved_name,
        }
    )
    return redirect(url_for("index", run_id=run_id))


@app.route("/depth_preview", methods=["POST"])
def preview_depth():
    run_id = request.form.get("run_id")
    if not run_id:
        flash("Session manquante pour les cartes de profondeur.")
        return redirect(url_for("index"))

    direction = request.form.get("direction", "forward")
    model_name = (request.form.get("model_name") or "").strip()
    try:
        depth_model_choice, depth_model_id, unet_model_path, resolved_name = resolve_depth_model_selection(
            model_name, DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH
        )
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index", run_id=run_id))

    if direction not in ("forward", "backward"):
        flash("Sens inconnu pour les cartes de profondeur.")
        return redirect(url_for("index", run_id=run_id))
    try:
        generate_depth_previews(
            run_id=run_id,
            direction=direction,
            model_name=resolved_name,
            depth_model_id=depth_model_id,
            unet_model_path=unet_model_path,
        )
    except Exception as exc:
        flash(f"Erreur pendant la generation des depth maps : {exc}")
        return redirect(url_for("index", run_id=run_id))

    run = RUNS.get(run_id)
    if run:
        run.setdefault("depth_settings", {}).update(
            {
                "model_choice": depth_model_choice,
                "model_id": depth_model_id,
                "unet_model_path": unet_model_path,
                "model_name": resolved_name,
            }
        )

    flash(f"Cartes de profondeur generees ({direction}, {resolved_name}).")
    return redirect(url_for("index", run_id=run_id))


@app.route("/evaluate_models", methods=["POST"])
def evaluate_models():
    run_id = request.form.get("run_id")
    if not run_id:
        flash("Session manquante pour l'evaluation des modeles.")
        return redirect(url_for("index"))

    direction = request.form.get("direction", "forward")
    if direction not in ("forward", "backward"):
        flash("Sens inconnu pour l'evaluation.")
        return redirect(url_for("index", run_id=run_id))

    model_name_a = (request.form.get("model_name_a") or "").strip()
    model_name_b = (request.form.get("model_name_b") or "").strip()
    if not model_name_a or not model_name_b:
        flash("Veuillez choisir deux modeles pour la comparaison.")
        return redirect(url_for("index", run_id=run_id))
    if model_name_a == model_name_b:
        flash("Veuillez choisir deux modeles differents.")
        return redirect(url_for("index", run_id=run_id))

    try:
        choice_a, model_id_a, unet_path_a, resolved_a = resolve_depth_model_selection(
            model_name_a, DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH
        )
        choice_b, model_id_b, unet_path_b, resolved_b = resolve_depth_model_selection(
            model_name_b, DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH
        )
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index", run_id=run_id))

    sources = get_sources_for_run(run_id, direction)
    if not sources:
        flash("Aucune image source disponible pour l'evaluation.")
        return redirect(url_for("index", run_id=run_id))

    model_key_a = build_depth_model_key(choice_a, resolved_a)
    model_key_b = build_depth_model_key(choice_b, resolved_b)
    label_a = build_depth_model_label(choice_a, resolved_a, model_id_a, unet_path_a)
    label_b = build_depth_model_label(choice_b, resolved_b, model_id_b, unet_path_b)

    per_image = []
    start = time.time()
    for path in sources:
        depth_a, _ = interpolate.get_depth_map(
            path,
            model_choice=choice_a,
            model_id=model_id_a,
            unet_model_path=unet_path_a,
        )
        depth_b, _ = interpolate.get_depth_map(
            path,
            model_choice=choice_b,
            model_id=model_id_b,
            unet_model_path=unet_path_b,
        )
        metrics = depth_metrics.compute_depth_metrics(
            depth_a,
            depth_b,
            min_valid_depth=depth_metrics.DEFAULT_MIN_VALID_DEPTH,
        )
        metrics["image"] = os.path.basename(path)
        per_image.append(metrics)

    summary = summarize_metrics(per_image)
    summary["min_valid_depth"] = depth_metrics.DEFAULT_MIN_VALID_DEPTH
    summary["duration_s"] = time.time() - start

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "direction": direction,
        "model_a": {
            "name": resolved_a,
            "choice": choice_a,
            "model_id": model_id_a,
            "unet_model_path": unet_path_a,
            "key": model_key_a,
            "label": label_a,
        },
        "model_b": {
            "name": resolved_b,
            "choice": choice_b,
            "model_id": model_id_b,
            "unet_model_path": unet_path_b,
            "key": model_key_b,
            "label": label_b,
        },
        "summary": summary,
        "per_image": per_image,
    }

    output_dir = metrics_output_dir(run_id, direction, model_key_a, model_key_b)
    output_path = os.path.join(output_dir, "metrics.json")
    output_txt_path = os.path.join(output_dir, "metrics.txt")
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        write_metrics_txt(output_txt_path, format_comparison_metrics_txt(payload))
    except OSError as exc:
        flash(f"Erreur lors de l'ecriture des metrics : {exc}")
        return redirect(url_for("index", run_id=run_id))

    flash(
        f"Evaluation terminee ({direction}) : {label_a} vs {label_b} sur {len(sources)} image(s)."
    )
    return redirect(url_for("index", run_id=run_id))


@app.route("/evaluate_models_groundtruth", methods=["POST"])
def evaluate_models_groundtruth():
    run_id = request.form.get("run_id")
    if not run_id:
        flash("Session manquante pour l'evaluation ground truth.")
        return redirect(url_for("index"))

    model_name_a = (request.form.get("model_name_a") or "").strip()
    model_name_b = (request.form.get("model_name_b") or "").strip()
    if not model_name_a or not model_name_b:
        flash("Veuillez choisir deux modeles pour la comparaison ground truth.")
        return redirect(url_for("index", run_id=run_id))
    if model_name_a == model_name_b:
        flash("Veuillez choisir deux modeles differents.")
        return redirect(url_for("index", run_id=run_id))

    pairs = list_ground_truth_pairs(GT_MAX_SAMPLES)
    if not pairs:
        flash("Fichiers ground truth manquants dans data_train_test/photos ou data_train_test/depth.")
        return redirect(url_for("index", run_id=run_id))
    if len(pairs) < GT_MAX_SAMPLES:
        flash(f"Seulement {len(pairs)} samples ground truth disponibles (attendu {GT_MAX_SAMPLES}).")

    try:
        choice_a, model_id_a, unet_path_a, resolved_a = resolve_depth_model_selection(
            model_name_a, DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH
        )
        choice_b, model_id_b, unet_path_b, resolved_b = resolve_depth_model_selection(
            model_name_b, DEPTH_MODEL_ID, DEFAULT_UNET_MODEL_PATH
        )
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index", run_id=run_id))

    model_key_a = build_depth_model_key(choice_a, resolved_a)
    model_key_b = build_depth_model_key(choice_b, resolved_b)
    label_a = build_depth_model_label(choice_a, resolved_a, model_id_a, unet_path_a)
    label_b = build_depth_model_label(choice_b, resolved_b, model_id_b, unet_path_b)
    output_dir = ground_truth_output_dir(run_id, model_key_a, model_key_b)

    sample_input_path = None
    depth_a_path = os.path.join(output_dir, "depth_model_a.jpg")
    depth_b_path = os.path.join(output_dir, "depth_model_b.jpg")
    depth_gt_path = os.path.join(output_dir, "depth_ground_truth.jpg")

    per_image = []
    metrics_a_list = []
    metrics_b_list = []
    gt_valid_ratios = []

    for idx, pair in enumerate(pairs):
        input_path = os.path.join(output_dir, f"input_{idx:03d}.jpg")
        input_path = _save_ground_truth_input(pair["photo"], input_path)

        depth_a, _ = interpolate.get_depth_map(
            input_path,
            model_choice=choice_a,
            model_id=model_id_a,
            unet_model_path=unet_path_a,
        )
        depth_b, _ = interpolate.get_depth_map(
            input_path,
            model_choice=choice_b,
            model_id=model_id_b,
            unet_model_path=unet_path_b,
        )
        gt_depth, gt_valid_ratio = load_ground_truth_depth(pair["depth"])

        metrics_a = depth_metrics.compute_depth_metrics(
            depth_a,
            gt_depth,
            min_valid_depth=depth_metrics.DEFAULT_MIN_VALID_DEPTH,
        )
        metrics_b = depth_metrics.compute_depth_metrics(
            depth_b,
            gt_depth,
            min_valid_depth=depth_metrics.DEFAULT_MIN_VALID_DEPTH,
        )
        metrics_a["valid_ratio_gt"] = gt_valid_ratio
        metrics_b["valid_ratio_gt"] = gt_valid_ratio

        metrics_a_list.append(metrics_a)
        metrics_b_list.append(metrics_b)
        gt_valid_ratios.append(gt_valid_ratio)

        per_image.append(
            {
                "image": os.path.relpath(pair["photo"], APP_ROOT).replace("\\", "/"),
                "depth": os.path.relpath(pair["depth"], APP_ROOT).replace("\\", "/"),
                "model_a": metrics_a,
                "model_b": metrics_b,
                "gt_valid_ratio": gt_valid_ratio,
            }
        )

        if idx == 0:
            sample_input_path = input_path
            cv2.imwrite(depth_a_path, render_depth_colormap(depth_a))
            cv2.imwrite(depth_b_path, render_depth_colormap(depth_b))
            cv2.imwrite(depth_gt_path, render_depth_colormap(gt_depth))

    summary_a = summarize_metrics(metrics_a_list)
    summary_b = summarize_metrics(metrics_b_list)
    gt_valid_ratio_avg = (
        sum(gt_valid_ratios) / len(gt_valid_ratios) if gt_valid_ratios else 0.0
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample": {
            "count": len(pairs),
            "photos_dir": "data_train_test/photos",
            "depth_dir": "data_train_test/depth",
            "max_samples": GT_MAX_SAMPLES,
        },
        "model_a": {
            "name": resolved_a,
            "choice": choice_a,
            "model_id": model_id_a,
            "unet_model_path": unet_path_a,
            "key": model_key_a,
            "label": label_a,
            "metrics": summary_a,
        },
        "model_b": {
            "name": resolved_b,
            "choice": choice_b,
            "model_id": model_id_b,
            "unet_model_path": unet_path_b,
            "key": model_key_b,
            "label": label_b,
            "metrics": summary_b,
        },
        "ground_truth": {
            "min_valid_depth": depth_metrics.DEFAULT_MIN_VALID_DEPTH,
            "valid_ratio": gt_valid_ratio_avg,
            "note": "Profondeur normalisee par min/max pour comparer la structure.",
        },
        "images": {
            "input": os.path.relpath(sample_input_path, BASE_OUTPUT).replace("\\", "/")
            if sample_input_path
            else None,
            "depth_a": os.path.relpath(depth_a_path, BASE_OUTPUT).replace("\\", "/"),
            "depth_b": os.path.relpath(depth_b_path, BASE_OUTPUT).replace("\\", "/"),
            "depth_gt": os.path.relpath(depth_gt_path, BASE_OUTPUT).replace("\\", "/"),
        },
        "per_image": per_image,
    }

    output_path = os.path.join(output_dir, "metrics.json")
    output_txt_path = os.path.join(output_dir, "metrics.txt")
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        write_metrics_txt(output_txt_path, format_groundtruth_metrics_txt(payload))
    except OSError as exc:
        flash(f"Erreur lors de l'ecriture des metrics ground truth : {exc}")
        return redirect(url_for("index", run_id=run_id))

    flash(f"Evaluation ground truth terminee : {label_a} vs {label_b} sur {len(pairs)} images.")
    return redirect(url_for("index", run_id=run_id))


@app.route("/make_video", methods=["POST"])
def make_video():
    run_id = request.form.get("run_id")
    direction = request.form.get("direction", "forward")
    speed_field = request.form.get("speed_m_s")

    if not run_id:
        flash("Session manquante pour la video.")
        return redirect(url_for("index"))

    if direction not in ("forward", "backward"):
        flash("Sens inconnu pour la video.")
        return redirect(url_for("index", run_id=run_id))

    frames_dir = os.path.join(BASE_OUTPUT, run_id, "frames", direction)
    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        flash("Aucune frame disponible pour ce sens.")
        return redirect(url_for("index", run_id=run_id))

    run = RUNS.get(run_id)
    path_info = run.get("paths", {}).get(direction) if run else {}
    sources = (path_info or {}).get("sources") or list_sources_from_disk(run_id, direction)
    meta = (path_info or {}).get("meta") or []

    segments = max(len(sources) - 1, 0)
    frames_per_official = None
    if run:
        frames_per_official = (run.get("interpolation", {}).get(direction) or {}).get("frames_per_official")
    if frames_per_official is None and segments > 0:
        frames_per_official = max(1, round((len(frame_paths) - 1) / segments))

    total_distance = sum(float(item.get("distance_from_prev_m") or 0.0) for item in meta) if meta else 0.0
    default_speed = VIDEO_TARGET_SPEED_M_S
    if run:
        default_speed = run.get("video_options", {}).get("speed_m_s", VIDEO_TARGET_SPEED_M_S)
    try:
        target_speed_m_s = float(speed_field) if speed_field is not None else default_speed
    except ValueError:
        target_speed_m_s = default_speed
    target_speed_m_s = max(0.5, min(50.0, target_speed_m_s))

    fps = compute_video_fps(total_distance, frames_per_official, segments, target_speed_m_s)

    videos_dir = ensure_dir(os.path.join(BASE_OUTPUT, run_id, "videos"))
    output_path = os.path.join(videos_dir, f"{direction}.mp4")

    try:
        log_console(f"[Video] Creation {direction}.mp4 avec {len(frame_paths)} frames @ {fps:.2f} fps (vitesse cible {target_speed_m_s:.1f} m/s).")
        write_video(frame_paths, output_path, fps)
    except Exception as exc:
        flash(f"Creation de la video impossible : {exc}")
        return redirect(url_for("index", run_id=run_id))

    if run is not None:
        run.setdefault("videos", {})[direction] = output_path
        run.setdefault("video_options", {})["speed_m_s"] = target_speed_m_s

    flash(f"Video {direction} creee ({len(frame_paths)} frames, {fps:.2f} fps, {target_speed_m_s:.1f} m/s).")
    return redirect(url_for("index", run_id=run_id))


@app.route("/output/<path:filename>")
def serve_output(filename):
    # Permet d'exposer les images generees sans copier dans static/
    return send_from_directory(BASE_OUTPUT, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
