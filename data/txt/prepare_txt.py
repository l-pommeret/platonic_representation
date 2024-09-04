import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import chardet

# Configuration
INPUT_FILE = "data/txt/all_tic_tac_toe_games.csv"
TRAIN_OUTPUT = "data/txt/train.bin"
VAL_OUTPUT = "data/txt/val.bin"
TRAIN_RATIO = 0.1
VECTOR_SIZE = 36
DTYPE = np.uint8

# Dictionnaire de conversion
META = {
    'stoi': {';': 0, ' ': 1, '0': 2, '1': 3, '2': 4, '3': 5, 'X': 6, 'O': 7, '/': 8, '-': 9, '\n': 10},
    'itos': {0: ';', 1: ' ', 2: '0', 3: '1', 4: '2', 5: '3', 6: 'X', 7: 'O', 8: '/', 9: '-', 10: '\n'}
}

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def load_csv_data(file_path, encoding):
    try:
        data = pd.read_csv(file_path, encoding=encoding, usecols=[0], header=None)
        data.columns = ['transcript']
        data['transcript'] = ';' + data['transcript']
        return data
    except UnicodeDecodeError:
        print(f"Échec de lecture avec l'encodage : {encoding}")
        return None

def process_line(line, meta=META, vector_size=VECTOR_SIZE):
    vector = np.ones(vector_size, dtype=DTYPE)
    for i, char in enumerate(str(line).strip()):
        if i >= vector_size:
            break
        vector[i] = meta['stoi'].get(char, 1)
    return vector

def process_data(data):
    return np.array([process_line(row['transcript']) for _, row in tqdm(data.iterrows(), total=len(data), desc="Traitement des lignes")])

def save_data(data, train_ratio, train_file, val_file):
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    
    train_data.tofile(train_file)
    val_data.tofile(val_file)
    
    print(f"Ensemble d'entraînement sauvegardé dans '{train_file}'")
    print(f"Ensemble de validation sauvegardé dans '{val_file}'")
    print(f"Nombre total d'exemples : {len(data)}")
    print(f"Nombre d'exemples d'entraînement : {len(train_data)}")
    print(f"Nombre d'exemples de validation : {len(val_data)}")

def load_and_print_batches(filename, start_batch=0, end_batch=10, batch_size=VECTOR_SIZE):
    data = np.fromfile(filename, dtype=DTYPE)
    total_batches = len(data) // batch_size
    start_batch = max(0, min(start_batch, total_batches - 1))
    end_batch = min(end_batch, total_batches)
    
    for i in range(start_batch, end_batch):
        batch = data[i*batch_size:(i+1)*batch_size]
        decoded = ''.join(META['itos'].get(token, ' ') for token in batch).rstrip()
        print(f"Batch {i+1}: {batch}")
        print(f"Décodé: {decoded}\n")

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Le fichier '{INPUT_FILE}' n'existe pas.")
    
    encoding = detect_file_encoding(INPUT_FILE)
    print(f"Encodage détecté : {encoding}")
    
    data = load_csv_data(INPUT_FILE, encoding)
    if data is None:
        raise ValueError("Impossible de lire le fichier avec l'encodage détecté.")
    
    print("Structure du DataFrame:")
    print(data.info())
    print("\nPremières lignes du DataFrame:")
    print(data.head())
    
    processed_data = process_data(data)
    save_data(processed_data, TRAIN_RATIO, TRAIN_OUTPUT, VAL_OUTPUT)
    
    print("\nAperçu des données d'entraînement:")
    load_and_print_batches(TRAIN_OUTPUT)

if __name__ == "__main__":
    main()