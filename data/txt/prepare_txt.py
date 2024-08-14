import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import chardet

# Dictionnaire de conversion fourni
meta = {
    'stoi': {';': 0, ' ': 1, '0': 2, '1': 3, '2': 4, '3': 5, 'X': 6, 'O': 7, '/': 8, '-': 9},
    'itos': {0: ';', 1: ' ', 2: '0', 3: '1', 4: '2', 5: '3', 6: 'X', 7: 'O', 8: '/', 9: '-'}
}
dtype = np.uint8  # 32 tokens seulement dans le vocabulaire des LLMs pour les échecs

# Chemin vers le fichier CSV local
local_file_path = "data/txt/all_tic_tac_toe_games.csv"

# Vérification de l'existence du fichier
if not os.path.exists(local_file_path):
    raise FileNotFoundError(f"Le fichier '{local_file_path}' n'existe pas.")

# Détection de l'encodage du fichier
with open(local_file_path, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    detected_encoding = result['encoding']
print(f"Encodage détecté : {detected_encoding}")

# Chargement du dataset à partir du fichier CSV local
encodings_to_try = [detected_encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
for encoding in encodings_to_try:
    try:
        # Lire seulement la première colonne du CSV
        data = pd.read_csv(local_file_path, encoding=encoding, usecols=[0], header=None)
        print(f"Lecture réussie avec l'encodage : {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Échec de lecture avec l'encodage : {encoding}")
else:
    raise ValueError("Impossible de lire le fichier avec les encodages essayés.")

# Renommer la colonne en 'transcript'
data.columns = ['transcript']

# Ajouter le point-virgule au début de chaque ligne
data['transcript'] = ';' + data['transcript']

# Afficher les informations sur le DataFrame pour le débogage
print("Structure du DataFrame:")
print(data.info())
print("\nPremières lignes du DataFrame:")
print(data.head())

def process_line(line, meta, vector_size=36):  # Augmenté à 36 pour inclure le point-virgule
    vector = np.zeros(vector_size, dtype=dtype)
    for i, char in enumerate(str(line).strip()):
        if i >= vector_size:
            break
        vector[i] = meta['stoi'].get(char, 1)  # Utiliser 1 (espace) si le caractère n'est pas trouvé
    return vector

# Traitement des données
batches = []
for _, row in tqdm(data.iterrows(), total=len(data), desc="Traitement des lignes"):
    text = row['transcript']
    batch = process_line(text, meta)
    batches.append(batch)

batches = np.array(batches)

# Sauvegarde des ensembles dans des fichiers binaires
train_ratio = 0.1  # 10% des parties possibles de tic tac toe pour l'entraînement pour voir le grok !
split_index = int(len(batches) * train_ratio)
train_batches = batches[:split_index]
val_batches = batches[split_index:]

train_batches.tofile("data/txt/train.bin")
val_batches.tofile("data/txt/val.bin")

print(f"Ensemble d'entraînement sauvegardé dans 'train.bin'")
print(f"Ensemble de validation sauvegardé dans 'val.bin'")

# Affichage des 50 premières lignes du CSV pour vérification
print("50 premières lignes du CSV:")
print(data.head(50))

# Informations sur les ensembles de données
print(f"\nNombre total d'exemples : {len(batches)}")
print(f"Nombre d'exemples d'entraînement : {len(train_batches)}")
print(f"Nombre d'exemples de validation : {len(val_batches)}")

def load_and_print_batches(filename, start_batch=0, end_batch=50, batch_size=36):  # Modifié à 36
    # Charger le fichier binaire
    data = np.fromfile(filename, dtype=np.uint8)
    
    # Calculer le nombre total de batches
    total_batches = len(data) // batch_size
    
    # Limiter l'intervalle de batches à afficher si nécessaire
    start_batch = max(0, start_batch)
    end_batch = min(end_batch, total_batches)
    
    # Afficher les batches dans l'intervalle spécifié
    for i in range(start_batch, end_batch):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch = data[batch_start:batch_end]
        print(f"Batch {i+1}: {batch}")
        
        # Convertir les tokens en caractères
        chars = [meta['itos'].get(token, ' ') for token in batch]
        print(f"Décodé: {''.join(chars)}")

# Exemple d'utilisation : afficher les batches de 1 à 10
load_and_print_batches("data/txt/train.bin", start_batch=0, end_batch=10)