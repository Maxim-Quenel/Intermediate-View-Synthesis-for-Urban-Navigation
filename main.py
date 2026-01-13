import os
import generate_frames  # Notre module pour l'API Street View
import interpolate      # Notre module pour l'IA et l'animation
from config_loader import DEFAULT_CONFIG_FILE, get_api_key, get_depth_model_id, get_unet_model_path, load_config

# --- CONFIGURATION ---
CONFIG = load_config()
API_KEY = get_api_key(CONFIG)
DEPTH_MODEL_ID = get_depth_model_id(CONFIG)
UNET_MODEL_PATH = get_unet_model_path(CONFIG)

ORIGIN_ADDRESS = "444 Chem. de Meaux, 93360 Neuilly-Plaisance, France"
INVERSER_SENS = True  # True = regarder derrière, False = devant
NUM_FRAMES = 30       # Nombre d'images pour la video (duree = NUM_FRAMES / fps)
DEPTH_MODEL_CHOICE = "depth-anything"

# --- GESTION DES DOSSIERS ---
BASE_OUTPUT = "street_view_project_output"
SOURCE_FOLDER = os.path.join(BASE_OUTPUT, "sources")
FRAMES_FOLDER = os.path.join(BASE_OUTPUT, "frames")

os.makedirs(SOURCE_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

def main():
    if not API_KEY:
        print(f"[ERREUR] Renseignez votre API_KEY dans {DEFAULT_CONFIG_FILE} (clé manquante).")
        return

    print("========================================")
    print("🚀 DÉMARRAGE DU PROJET STREET VIEW AI")
    print("========================================")

    # ÉTAPE 1 : Récupération des images sources
    print(f"\n📍 Adresse : {ORIGIN_ADDRESS}")
    img_a, img_b = generate_frames.fetch_source_images(
        address=ORIGIN_ADDRESS,
        api_key=API_KEY,
        output_folder=SOURCE_FOLDER,
        inverser_sens=INVERSER_SENS
    )

    if not img_a or not img_b:
        print("❌ Arrêt du script : Impossible de récupérer les images.")
        return

    print("\n📸 Images sources récupérées avec succès :")
    print(f"   A: {img_a}")
    print(f"   B: {img_b}")

    # ÉTAPE 2 : Génération de l'animation
    print("\n🎨 Lancement de l'interpolation IA...")
    success = interpolate.process_interpolation(
        img_a_path=img_a,
        img_b_path=img_b,
        output_frames_folder=FRAMES_FOLDER,
        num_frames=NUM_FRAMES,
        depth_model_choice=DEPTH_MODEL_CHOICE,
        depth_model_id=DEPTH_MODEL_ID,
        unet_model_path=UNET_MODEL_PATH,
    )

    if success:
        print("\n�
 TERMINÉ ! Vos frames sont dans :", FRAMES_FOLDER)
        print("Pour créer la vidéo MP4, lancez cette commande dans le terminal :")
        print(f'ffmpeg -framerate 10 -i "{FRAMES_FOLDER}/frame_%03d.jpg" -c:v libx264 -pix_fmt yuv420p output.mp4')

if __name__ == "__main__":
    main()
