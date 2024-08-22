import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/gif/files"  # Dossier contenant les GIFs
TRAIN_OUTPUT = "data/gif/train.bin"
VAL_OUTPUT = "data/gif/val.bin"
TRAIN_RATIO = 0.9
FRAME_SIZE = 3  # 3x3 pixels par frame
NUM_FRAMES = 9  # Nombre de frames par GIF
VECTOR_SIZE = FRAME_SIZE * FRAME_SIZE * NUM_FRAMES + 1  # +1 pour le token de début
DTYPE = np.uint8

# Valeurs de couleur fixes
EMPTY = 128  # Gris
O = 0  # Noir
X = 255  # Blanc

# Dictionnaire de conversion
META = {
    'stoi': {'b': 0, 'n': 1, 'g': 2, ';': 3},
    'itos': {0: 'b', 1: 'n', 2: 'g', 3: ';'}
}

def load_and_process_gif(gif_path):
    vector = np.zeros(VECTOR_SIZE, dtype=DTYPE)
    vector[0] = META['stoi'][';']  # Ajouter le token de début
    
    with Image.open(gif_path) as gif:
        for frame_index in range(NUM_FRAMES):
            try:
                gif.seek(frame_index)
                frame = gif.convert('L').resize((FRAME_SIZE, FRAME_SIZE))
                data = np.array(frame)
            except EOFError:
                # Si on atteint la fin du GIF, on remplit avec des pixels vides (gris)
                data = np.full((FRAME_SIZE, FRAME_SIZE), EMPTY)
            
            # Convertir les valeurs de pixel en tokens
            data = np.where(data == X, META['stoi']['b'],
                    np.where(data == O, META['stoi']['n'],
                    np.where(data == EMPTY, META['stoi']['g'], META['stoi']['g'])))
            
            # Ajouter les données de la frame au vecteur
            start_index = 1 + frame_index * FRAME_SIZE * FRAME_SIZE
            end_index = start_index + FRAME_SIZE * FRAME_SIZE
            vector[start_index:end_index] = data.flatten()
    
    return vector

def process_data(input_dir):
    all_data = []
    for filename in tqdm(os.listdir(input_dir), desc="Traitement des GIFs"):
        if filename.endswith('.gif'):
            file_path = os.path.join(input_dir, filename)
            try:
                vector = load_and_process_gif(file_path)
                all_data.append(vector)
            except Exception as e:
                print(f"Erreur lors du traitement de {filename}: {str(e)}")
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
        
        for frame in range(NUM_FRAMES):
            print(f"Frame {frame + 1}:")
            start_index = 1 + frame * FRAME_SIZE * FRAME_SIZE
            for row in range(FRAME_SIZE):
                row_start = start_index + row * FRAME_SIZE
                row_end = row_start + FRAME_SIZE
                print(''.join(META['itos'][token] for token in batch[row_start:row_end]))
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