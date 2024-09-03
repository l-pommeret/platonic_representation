import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import os
import traceback
import random
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from PIL import Image

from extract_features import load_model
from data.gif.prepare_gif import load_and_process_gif

# Configuration
FRAME_SIZE = 3
NUM_FRAMES = 9
VECTOR_SIZE = FRAME_SIZE * FRAME_SIZE * NUM_FRAMES + 1
META = {
    'stoi': {'b': 0, 'n': 1, 'g': 2, ';': 3},
    'itos': {0: 'b', 1: 'n', 2: 'g', 3: ';'}
}

def get_probe_points(model):
    """Dynamically generate probe points based on model architecture."""
    probe_points = ['embedding']
    
    num_layers = len(model.transformer.h)
    for layer in range(num_layers):
        probe_points.extend([
            f'layer{layer}_pre_attn_norm',
            f'layer{layer}_attn',
            f'layer{layer}_post_attn',
            f'layer{layer}_pre_ffn_norm',
            f'layer{layer}_mlp_fc',
            f'layer{layer}_mlp_proj'
        ])
    
    probe_points.extend(['final_ln', 'lm_head'])
    return probe_points

def extract_activations_all_points(model, gif_paths, device, batch_size=4, chunk_size=100):
    """Extract activations for all possible points in the model."""
    model.eval()
    
    probe_points = get_probe_points(model)
    
    all_activations = {point: [] for point in probe_points}
    
    for chunk_start in tqdm(range(0, len(gif_paths), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(gif_paths))
        chunk_gifs = gif_paths[chunk_start:chunk_end]
        
        chunk_activations = {point: [] for point in probe_points}
        
        for i in range(0, len(chunk_gifs), batch_size):
            batch_gifs = chunk_gifs[i:i+batch_size]
            try:
                batch_data = [load_and_process_gif(gif_path) for gif_path in batch_gifs]
                
                batch_data_np = np.array(batch_data, dtype=np.int64)
                batch_tensor = torch.from_numpy(batch_data_np).to(device)
                
                with torch.no_grad():
                    # Embedding
                    embedding = model.transformer.wte(batch_tensor)
                    chunk_activations['embedding'].append(embedding.cpu())
                    
                    x = model.transformer.drop(embedding)
                    
                    # Process all layers
                    for layer_idx, block in enumerate(model.transformer.h):
                        ln1_out = block.ln_1(x)
                        chunk_activations[f'layer{layer_idx}_pre_attn_norm'].append(ln1_out.cpu())
                        
                        attn_output = block.attn(ln1_out)
                        chunk_activations[f'layer{layer_idx}_attn'].append(attn_output.cpu())
                        
                        x = x + attn_output
                        chunk_activations[f'layer{layer_idx}_post_attn'].append(x.cpu())
                        
                        ln2_out = block.ln_2(x)
                        chunk_activations[f'layer{layer_idx}_pre_ffn_norm'].append(ln2_out.cpu())
                        
                        # MLP layers
                        mlp_fc = block.mlp.c_fc(ln2_out)
                        chunk_activations[f'layer{layer_idx}_mlp_fc'].append(mlp_fc.cpu())
                        
                        mlp_output = block.mlp(ln2_out)
                        chunk_activations[f'layer{layer_idx}_mlp_proj'].append(mlp_output.cpu())
                        
                        x = x + mlp_output
                    
                    # Final layer norm
                    x = model.transformer.ln_f(x)
                    chunk_activations['final_ln'].append(x.cpu())
                    
                    # Language model head
                    lm_output = model.lm_head(x)
                    chunk_activations['lm_head'].append(lm_output.cpu())
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                print(traceback.format_exc())
                continue
        
        # Concatenate and store chunk activations
        for point in probe_points:
            if chunk_activations[point]:
                all_activations[point].append(torch.cat(chunk_activations[point], dim=0))
            else:
                print(f"Warning: No activations for {point} in this chunk")
    
    # Concatenate all chunks
    for point in probe_points:
        if all_activations[point]:
            all_activations[point] = torch.cat(all_activations[point], dim=0).numpy()
        else:
            print(f"Warning: No activations for {point} across all chunks")
            all_activations[point] = np.array([])
    
    return all_activations

def prepare_labels(gif_paths):
    """Prepare labels for the tic-tac-toe board states."""
    labels = []
    for gif_path in gif_paths:
        board = ['-'] * 9
        with Image.open(gif_path) as gif:
            for frame in range(gif.n_frames):
                gif.seek(frame)
                frame_img = gif.convert('L')
                for i in range(3):
                    for j in range(3):
                        pixel = frame_img.getpixel((i, j))
                        if pixel == 0:  # Black (O)
                            board[i + j * 3] = 'O'
                        elif pixel == 255:  # White (X)
                            board[i + j * 3] = 'X'
        labels.append(board)
    return np.array(labels)

def train_and_evaluate_probing_classifiers(activations, labels, max_iter=1000):
    """Train and evaluate probing classifiers for each board position using LinearSVC."""
    results = []
    n_splits = 5
    
    if activations.ndim == 3:
        activations = activations.mean(axis=1)
    
    # Normalisation des activations
    scaler = StandardScaler()
    activations_normalized = scaler.fit_transform(activations)
    
    for position in range(9):
        y = labels[:, position]
        base_clf = LinearSVC(max_iter=max_iter, dual=False)
        clf = OneVsRestClassifier(base_clf)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_metrics = []
        val_metrics = []
        
        for train_index, val_index in kf.split(activations_normalized):
            X_train, X_val = activations_normalized[train_index], activations_normalized[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            clf.fit(X_train, y_train)
            
            train_pred = clf.predict(X_train)
            val_pred = clf.predict(X_val)
            
            train_metrics.append({
                'accuracy': accuracy_score(y_train, train_pred)
            })
            val_metrics.append({
                'accuracy': accuracy_score(y_val, val_pred)
            })
        
        results.append({
            'position': position,
            'train_accuracy': np.mean([m['accuracy'] for m in train_metrics]),
            'val_accuracy': np.mean([m['accuracy'] for m in val_metrics])
        })
    
    return results

def process_all_points(model, gif_paths, device, labels, max_iter=1000):
    """Process all probe points: extract activations and train probing classifiers."""
    try:
        print("Starting to process all probe points")
        all_activations = extract_activations_all_points(model, gif_paths, device)
        
        all_results = {}
        for point, activations in all_activations.items():
            if activations.size == 0:
                print(f"Skipping {point} due to empty activations")
                continue
            
            print(f"Processing probe point: {point}")
            print(f"Activations shape: {activations.shape}")
            
            results = train_and_evaluate_probing_classifiers(activations, labels, max_iter=max_iter)
            
            print(f"Probe point: {point}")
            avg_train_accuracy = 0
            avg_val_accuracy = 0
            for pos_result in results:
                print(f"  Position {pos_result['position'] + 1}:")
                print(f"    Train Accuracy: {pos_result['train_accuracy']:.4f}")
                print(f"    Validation Accuracy: {pos_result['val_accuracy']:.4f}")
                avg_train_accuracy += pos_result['train_accuracy']
                avg_val_accuracy += pos_result['val_accuracy']
            
            avg_train_accuracy /= 9
            avg_val_accuracy /= 9
            print(f"\n{point} Summary:")
            print(f"  Average Train Accuracy: {avg_train_accuracy:.4f}")
            print(f"  Average Validation Accuracy: {avg_val_accuracy:.4f}")
            print()
            
            all_results[point] = results
            
            with open(f"probing_results_gif_{point}.pkl", 'wb') as f:
                pickle.dump(results, f)
        
        return all_results
    except Exception as e:
        print(f"Error processing probe points: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_graphs(all_results):
    """Generate and save graphs for probing results."""
    probe_points = list(all_results.keys())
    positions = list(range(1, 10))

    # Plotting average accuracy across probe points
    plt.figure(figsize=(15, 8))
    avg_accuracies = [np.mean([result['val_accuracy'] for result in results]) for results in all_results.values()]

    # Create custom x-axis labels
    x_labels = []
    x_ticks = []
    for i, point in enumerate(probe_points):
        if 'layer' in point:
            layer, component = point.split('_', 1)
            x_labels.append(f"{layer}\n{component}")
        else:
            x_labels.append(point)
        x_ticks.append(i)

    plt.plot(x_ticks, avg_accuracies, marker='o')
    plt.xlabel('Probe Point')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Average Accuracy Across Probe Points (Linear Probing) - GIF')
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
    plt.ylim(0.45, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assets/gif_accuracy_across_probe_points.png')
    plt.close()

    # Plotting accuracy across positions for each probe point
    plt.figure(figsize=(15, 8))
    for i, (point, results) in enumerate(all_results.items()):
        accuracies = [result['val_accuracy'] for result in results]
        label = point.replace('_', ' ').title()
        plt.plot(positions, accuracies, marker='o', label=label)
    plt.xlabel('Board Position')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Across Board Positions for Each Probe Point (Linear Probing) - GIF')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('assets/gif_accuracy_across_positions_all_points.png')
    plt.close()

    # Heatmap of accuracies
    accuracies = np.array([[result['val_accuracy'] for result in results] for results in all_results.values()])
    plt.figure(figsize=(15, 10))
    plt.imshow(accuracies, cmap='viridis', aspect='auto')
    plt.colorbar(label='Validation Accuracy')
    plt.xlabel('Board Position')
    plt.ylabel('Probe Point')
    plt.title('Heatmap of Accuracies Across Probe Points and Board Positions - GIF')
    plt.yticks(range(len(probe_points)), x_labels)
    plt.xticks(range(9), positions)
    plt.tight_layout()
    plt.savefig('assets/gif_accuracy_heatmap.png')
    plt.close()

    print("Graphs have been saved in the assets directory.")

    # Generate CSV file
    with open('gif_validation_accuracy.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Probe Point'] + [f'Position {i}' for i in range(1, 10)]
        writer.writerow(header)

        for point, results in all_results.items():
            row = [point] + [result['val_accuracy'] for result in results]
            writer.writerow(row)

    print("CSV file 'gif_validation_accuracy.csv' has been generated.")

    # Generate heatmaps for each layer
    for point, results in all_results.items():
        if 'layer' in point:
            accuracies = [result['val_accuracy'] for result in results]
            accuracies_matrix = np.array(accuracies).reshape(3, 3)

            plt.figure(figsize=(8, 6))
            sns.heatmap(accuracies_matrix, annot=True, cmap='viridis', vmin=0, vmax=1)
            plt.title(f'Tic-Tac-Toe Board Heatmap - {point} (GIF)')
            plt.savefig(f'assets/gif_ttt_heatmap_{point}.png')
            plt.close()

    print("Tic-Tac-Toe board heatmaps have been saved in the assets directory.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("out-gif-models/ckpt_iter_4000.pt").to(device)

    # Load and sample GIFs
    gif_dir = "data/gif/files"
    gif_paths = [os.path.join(gif_dir, f) for f in os.listdir(gif_dir) if f.endswith('.gif')]

    random.shuffle(gif_paths)
    sampled_gifs = gif_paths[:3000]  # Sample 3000 GIFs for processing

    labels = prepare_labels(sampled_gifs)
    all_results = process_all_points(model, sampled_gifs, device, labels)
    
    if all_results:
        generate_graphs(all_results)

if __name__ == "__main__":
    main()