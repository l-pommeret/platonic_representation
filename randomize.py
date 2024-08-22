import torch
import torch.nn as nn
import os

def randomize_transformer_weights(model_path, output_path):
    # Vérifier si MPS est disponible
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Utilisation du dispositif: {device}")

    # Charger le modèle
    checkpoint = torch.load(model_path, map_location=device)

    # Fonction pour initialiser les poids de manière complètement aléatoire
    def init_weights(tensor):
        if tensor.ndim > 1:
            nn.init.normal_(tensor, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(tensor)

    # Parcourir le dictionnaire et randomiser les poids
    for key in checkpoint:
        if isinstance(checkpoint[key], torch.Tensor):
            init_weights(checkpoint[key])
        elif isinstance(checkpoint[key], dict):
            for subkey in checkpoint[key]:
                if isinstance(checkpoint[key][subkey], torch.Tensor):
                    init_weights(checkpoint[key][subkey])

    # Sauvegarder le modèle avec les poids randomisés
    torch.save(checkpoint, output_path)
    print(f"Modèle avec poids complètement randomisés sauvegardé dans {output_path}")

# Utilisation du script
model_path = "out-gif-models/ckpt_iter_0.pt"
output_path = "out-gif-models/ckpt_iter_0_rand.pt"
randomize_transformer_weights(model_path, output_path)