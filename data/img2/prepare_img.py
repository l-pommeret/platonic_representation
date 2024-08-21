import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/img/files"  # Dossier contenant les images
TRAIN_OUTPUT = "data/img/train.bin"
VAL_OUTPUT = "data/img/val.bin"
TRAIN_RATIO = 0.1
IMAGE_SIZE = 9  # 9x9 pixels
VECTOR_SIZE = IMAGE_SIZE * IMAGE_SIZE + 1  # +1 pour le token de début
DTYPE = np.uint8

# Valeurs de couleur fixes
EMPTY = 128  # Gris
O = 0        # Noir
X = 255      # Blanc

# Dictionnaire de conversion
META = {
    'stoi': {'b': 0, 'n': 1, 'g': 2, ';': 3},
    'itos': {0: 'b', 1: 'n', 2: 'g', 3: ';'}
}

def load_and_process_image(img):
    img = img.convert('L')  # Convertir en niveaux de gris
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # Redimensionner à 9x9
    data = np.array(img)
    
    # Convertir les valeurs de pixel en tokens
    data = np.where(data == X, META['stoi']['b'],
            np.where(data == O, META['stoi']['n'],
            np.where(data == EMPTY, META['stoi']['g'], META['stoi']['g'])))
    
    # Créer le vecteur avec le token de début
    vector = np.zeros(VECTOR_SIZE, dtype=DTYPE)
    vector[0] = META['stoi'][';']  # Ajouter le token de début
    
    # Parcours normal (ligne par ligne, de haut en bas, de gauche à droite)
    vector[1:] = data.flatten()
    
    return vector

def process_data(input_dir):
    all_data = []
    for filename in tqdm(os.listdir(input_dir), desc="Traitement des images"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_dir, filename)
            with Image.open(file_path) as img:
                vector = load_and_process_image(img)
            all_data.append(vector)
    return np.array(all_data)

def save_data(data, train_ratio, train_file, val_file):
    np.random.shuffle(data)
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
        print(f"Batch {i+1}: {batch}")
        print("Décodé:")
        print(f"Token de début: {META['itos'][batch[0]]}")
        for row in range(IMAGE_SIZE):
            print(''.join(META['itos'][token] for token in batch[1+row*IMAGE_SIZE:1+(row+1)*IMAGE_SIZE]))
        print()

def main():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Le dossier '{INPUT_DIR}' n'existe pas.")
    
    processed_data = process_data(INPUT_DIR)
    save_data(processed_data, TRAIN_RATIO, TRAIN_OUTPUT, VAL_OUTPUT)
    
    print("\nAperçu des données d'entraînement:")
    load_and_print_batches(TRAIN_OUTPUT)

if __name__ == "__main__":
    main()