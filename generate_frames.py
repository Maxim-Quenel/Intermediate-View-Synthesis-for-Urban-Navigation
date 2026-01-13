import math
import os
import polyline
import requests


# ============================================================
# OUTILS MATHEMATIQUES
# ============================================================

def calculate_heading(p1, p2):
    """Calcule l'angle (0-360) de P1 vers P2."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_destination_point(lat, lon, distance_meters, heading):
    """Calcule le point GPS situe X metres devant."""
    R = 6378137
    d = distance_meters
    brng = math.radians(heading)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d / R)
        + math.cos(lat1) * math.sin(d / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d / R) * math.cos(lat1),
        math.cos(d / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def get_real_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance en mètres entre deux points GPS (Formule de Haversine)."""
    R = 6378137  # Rayon de la Terre en mètres
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ============================================================
# LOGIQUE API GOOGLE
# ============================================================

def get_base_heading(address, api_key):
    """Recupere l'angle par defaut de la route."""
    geo_resp = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": address, "key": api_key},
    ).json()
    if geo_resp.get("status") != "OK":
        return None, None
    lat = geo_resp["results"][0]["geometry"]["location"]["lat"]
    lng = geo_resp["results"][0]["geometry"]["location"]["lng"]

    # On trace une mini-route vers le Nord pour avoir l'axe
    fake_dest = f"{lat + 0.005},{lng + 0.005}"
    dir_resp = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params={
            "origin": address,
            "destination": fake_dest,
            "mode": "driving",
            "key": api_key,
        },
    ).json()
    if dir_resp.get("status") != "OK":
        return None, None

    points = polyline.decode(dir_resp["routes"][0]["overview_polyline"]["points"])
    if len(points) < 2:
        return None, None

    p_start = points[0]
    heading = calculate_heading(points[0], points[1])

    return p_start, heading


def get_pano_data(lat, lng, api_key):
    """
    Récupère l'ID ET les coordonnées réelles du panorama le plus proche.
    """
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "radius": 50, "source": "outdoor", "key": api_key}

    resp = requests.get(url, params=params).json()

    if resp.get("status") == "OK":
        location = resp.get("location", {})
        return {
            "pano_id": resp.get("pano_id"),
            "lat": location.get("lat"),
            "lng": location.get("lng")
        }
    return None


def find_next_pano(start_lat, start_lng, start_id, heading, api_key):
    """
    Avance dans la direction donnee jusqu'a changer de photo.
    Retourne l'ID, la DISTANCE REELLE entre les deux points GPS, et la nouvelle position.
    """
    current_dist = 1  # Commence à chercher à 1m
    max_dist = 50
    step = 1          # Cherche mètre par mètre pour ne rien rater

    while current_dist <= max_dist:
        # Point de test théorique sur la route
        test_lat, test_lng = calculate_destination_point(start_lat, start_lng, current_dist, heading)
        
        # Interrogation API
        data = get_pano_data(test_lat, test_lng, api_key)

        if data:
            found_id = data["pano_id"]
            found_lat = data["lat"]
            found_lng = data["lng"]

            if found_id and found_id != start_id:
                # Calcul de la distance réelle exacte entre la photo A et la photo B
                real_distance = get_real_distance(start_lat, start_lng, found_lat, found_lng)
                return found_id, real_distance, found_lat, found_lng

        current_dist += step
        
    return None, None, None, None


def download_view(pano_id, heading, filename, api_key):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "pano": pano_id,
        "heading": heading,
        "pitch": "0",
        "fov": "90",
        "key": api_key,
    }
    r = requests.get(url, params=params)
    with open(filename, "wb") as f:
        f.write(r.content)


# ============================================================
# FONCTION PRINCIPALE APPELEE PAR MAIN.PY
# ============================================================

def fetch_source_images(address, api_key, output_folder, inverser_sens=False, num_sources=2, return_meta=False):
    """Telecharge une sequence de panoramas Street View dans un meme sens."""
    print(">>> [Generate Frames] 1. Calcul de la trajectoire...")
    
    # 1. Trouve la direction de la route
    start_point_route, heading = get_base_heading(address, api_key)

    if not start_point_route:
        print("Erreur : Impossible de trouver la route.")
        return []

    if inverser_sens:
        print(f"    Inversion demandee (angle original : {heading:.1f} deg)")
        heading = (heading + 180) % 360
        print(f"    Nouvel angle (sens oppose) : {heading:.1f} deg")
    else:
        print(f"    Angle conserve : {heading:.1f} deg")

    # 2. Récupère les données précises du PREMIER panorama
    start_data = get_pano_data(start_point_route[0], start_point_route[1], api_key)
    
    if not start_data:
        print("Erreur : Impossible de recuperer le premier panorama.")
        return []

    pano_id = start_data["pano_id"]
    current_lat = start_data["lat"] # Position réelle de la caméra Google
    current_lng = start_data["lng"]

    sources = []
    meta = []
    last_dist = 0.0

    for idx in range(num_sources):
        filename = os.path.join(output_folder, f"source_{idx:02d}.jpg")
        
        # Téléchargement
        download_view(pano_id, heading, filename, api_key)
        sources.append(filename)
        
        # Métadonnées
        meta.append(
            {
                "lat": current_lat,
                "lng": current_lng,
                "pano_id": pano_id,
                "heading": heading,
                "distance_from_prev_m": last_dist,
            }
        )

        # Recherche de la prochaine image
        next_id, real_dist, next_lat, next_lng = find_next_pano(current_lat, current_lng, pano_id, heading, api_key)
        
        if not next_id:
            print(f"    Arret : seulement {len(sources)} panoramas trouves (demande initiale : {num_sources}).")
            break

        print(f"    ID suivant : {next_id} (distance réelle : {real_dist:.2f}m)")
        
        # Mise à jour pour la prochaine boucle
        pano_id = next_id
        current_lat, current_lng = next_lat, next_lng
        last_dist = float(real_dist or 0.0)

    if len(sources) < 2:
        print("Erreur : moins de 2 panoramas disponibles.")
        return []

    if return_meta:
        return sources, meta
    return sources