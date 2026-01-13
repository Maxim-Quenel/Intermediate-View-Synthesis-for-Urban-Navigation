# Structure du projet

## Vue d'ensemble
Ce projet genere des vues intermediaires a partir de panoramas Street View pour simuler un mouvement continu. Il combine collecte d'images via l'API Google, estimation de profondeur monoculaire, reprojection 3D avec softmax splatting, fusion bidirectionnelle et inpainting, puis export en video.

## Architecture (modules principaux)
```
projet/
|-- app.py : serveur Flask, orchestration des runs, UI, previsualisation, evaluation, generation de frames et videos
|-- generate_frames.py : scraping Street View (geocode, directions, metadata, download), calculs d'orientation et distances GPS
|-- depth_models.py : inference profondeur (Depth Anything V2 via Transformers, UNet local), normalisation et colormap
|-- interpolate.py : warping 3D, softmax splatting, fusion forward/backward, inpainting LaMa ou OpenCV, generation de frames
|-- depth_metrics.py : MAE, RMSE, Pearson r, Edge F1, valid_ratio pour comparer des cartes de profondeur
|-- config_loader.py : charge config.txt et variables d'environnement (API key, model id, UNet path)
|-- templates/
|   |-- run.html : interface Web (run, previews, galerie, metrics, video)
|   `-- index.html : interface Web (landing)
|-- static/
|   `-- leaflet/ : assets front pour la carte
|-- train_unet.ipynb : entrainement UNet sur Cityscapes Depth and Segmentation
|-- data_train_test/ : batch GT (photos/*.npy, depth/*.npy) pour test ground truth
`-- street_view_project_output/ : sorties par run (sources, frames, previews, metrics, videos)
```

## Technologies et methodes
- Python + Flask pour le serveur et l'orchestration.
- Requests + Google Maps APIs (Geocode, Directions, Street View Metadata/Image) pour le scraping.
- Polyline pour decoder l'itineraire et calculer le heading.
- Torch + Transformers pour Depth Anything V2 (depth-anything/Depth-Anything-V2-Large-hf).
- UNet custom (PyTorch) pour profondeur supervissee (Cityscapes Depth and Segmentation).
- OpenCV + NumPy + PIL pour I/O image, post-processing et video.
- Softmax splatting (warping differentiable) pour la reprojection 3D et la gestion d'occlusions.
- Fusion bidirectionnelle (forward/backward) avec regles d'occlusion + detection de changement visuel.
- Inpainting LaMa (simple-lama-inpainting) avec fallback OpenCV (Navier-Stokes).
- Metriques de profondeur (MAE, RMSE, Pearson r, Edge F1, valid ratio).

## Liens avec SOTA
- Depth Anything V2 : utilise pour l'estimation de profondeur monoculaire (inference via Transformers).
- Softmax splatting : utilise pour le warping differentiable et la gestion des occlusions.
- LaMa : utilise pour l'inpainting des disocclusions (fallback OpenCV si absent).

## Flux de donnees (scraping -> video)
1) Saisie utilisateur (adresse, API key, nb de panoramas) dans l'interface Flask.
2) generate_frames.fetch_source_images : geocode -> coordonnees de depart; directions + polyline -> heading; Street View metadata -> pano_id et position reelle; avance metre par metre pour trouver le pano suivant; download des images (640x640, FOV 90, pitch 0) + meta (distance Haversine).
3) app.py stocke les sources et meta dans street_view_project_output/run-*/{forward,backward}/sources.
4) Interpolation par segment (A->B) via interpolate.process_interpolation : predict_depth_map (Depth Anything ou UNet) + normalisation (seuil ciel 0.03); conversion en pseudo-profondeur metrique (z = 1/(d+0.01), ciel a grande distance); warping 3D (pinhole FOV 90) + flow 2D + softmax splatting; fusion des warps A/B (blend temporel, occlusion threshold 0.5, change RGB > 30); inpainting des trous (LaMa ou OpenCV); ecriture des frames, insertion des frames officielles aux bornes des segments.
5) make_video : calcule le fps a partir de la distance GPS et d'une vitesse cible (clamp 8-60), puis assemble les frames en MP4 via OpenCV VideoWriter.
6) Optionnel : generation de previews depth, evaluation inter-modeles et comparaison ground truth.

## Organisation des sorties (exemple)
```
street_view_project_output/
  run-YYYYMMDD-HHMMSS/
    forward/sources/
    backward/sources/
    frames/forward/
    frames/backward/
    depth_previews/forward/<model_key>/
    depth_previews/backward/<model_key>/
    model_metrics/forward/<modelA__vs__modelB>/
    model_metrics/ground_truth/<modelA__vs__modelB>/
    videos/forward.mp4
    videos/backward.mp4
```

## Sequence de donnees (resume court)
Saisie adresse -> scraping Street View -> sources + meta -> depth estimation -> warping 3D + fusion -> inpainting -> frames -> video.
