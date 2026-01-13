import os
from functools import lru_cache


# Fichier de configuration par defaut (modifiable via la variable d'environnement CONFIG_FILE)
DEFAULT_CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.txt")
# Valeur par defaut pour le modele de profondeur si aucune entree n'est definie dans le fichier.
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


def _parse_config_line(line):
    """Parse une ligne de la forme CLE=VALEUR et ignore les commentaires/blancs."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None
    if "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


def _read_config(path):
    config = {}
    if not os.path.exists(path):
        return config
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            key, value = _parse_config_line(raw_line)
            if key:
                config[key] = value
    return config


@lru_cache(maxsize=1)
def load_config(path=None):
    """Charge et met en cache le contenu du fichier de configuration."""
    path = path or DEFAULT_CONFIG_FILE
    return _read_config(path)


def get_api_key(config=None):
    """Retourne la cle API Google en priorisant l'environnement puis le fichier texte."""
    cfg = config or load_config()
    return (os.environ.get("GOOGLE_API_KEY") or cfg.get("API_KEY") or "").strip()


def get_depth_model_id(config=None):
    """Retourne l'identifiant du modele de profondeur."""
    cfg = config or load_config()
    return (
        cfg.get("DEPTH_MODEL_ID")
        or os.environ.get("DEPTH_MODEL_ID")
        or DEFAULT_DEPTH_MODEL
    ).strip()


def get_unet_model_path(config=None):
    """Retourne le chemin vers les poids UNet si defini."""
    cfg = config or load_config()
    return (
        cfg.get("UNET_MODEL_PATH")
        or os.environ.get("UNET_MODEL_PATH")
        or ""
    ).strip()


def get_unet_use_batchnorm(config=None):
    """Retourne True si UNET_USE_BATCHNORM est active."""
    cfg = config or load_config()
    raw = (
        cfg.get("UNET_USE_BATCHNORM")
        or os.environ.get("UNET_USE_BATCHNORM")
        or ""
    )
    return raw.strip().lower() in {"1", "true", "yes", "on"}
