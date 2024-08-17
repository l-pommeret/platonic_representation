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

from extract_features import load_model, tokenize_and_pad

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

def extract_activations_all_points(model, games, tokenizer, device, batch_size=4, chunk_size=100):
    """Extract activations for all possible points in the model."""
    model.eval()
    max_length = model.config.block_size

    probe_points = get_probe_points(model)

    all_activations = {point: [] for point in probe_points}

    for chunk_start in tqdm(range(0, len(games), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(games))
        chunk_games = games[chunk_start:chunk_end]

        chunk_activations = {point: [] for point in probe_points}

        for i in range(0, len(chunk_games), batch_size):
            batch = chunk_games[i:i+batch_size]
            input_ids = tokenize_and_pad(batch, tokenizer, max_length).to(device)

            with torch.no_grad():
                # Embedding
                embedding = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(1), device=device))
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

def prepare_labels(games):
    """Prepare labels for the tic-tac-toe board states."""
    labels = []
    for game in games:
        board = ['-'] * 9
        moves = game[1:].split()
        for i, move in enumerate(moves):
            player = 'X' if i % 2 == 0 else 'O'
            position = int(move[1]) - 1 + 3 * (int(move[2]) - 1)
            board[position] = player
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

def process_all_points(model, games, tokenizer, device, labels, max_iter=1000):
    """Process all probe points: extract activations and train probing classifiers."""
    try:
        print("Starting to process all probe points")
        all_activations = extract_activations_all_points(model, games, tokenizer, device)

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

            with open(f"probing_results_{point}.pkl", 'wb') as f:
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
    plt.title('Average Accuracy Across Probe Points (Linear Probing)')
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assets/txt_accuracy_across_probe_points.png')
    plt.close()

    # Plotting accuracy across positions for each probe point
    plt.figure(figsize=(15, 8))
    for i, (point, results) in enumerate(all_results.items()):
        accuracies = [result['val_accuracy'] for result in results]
        label = point.replace('_', ' ').title()
        plt.plot(positions, accuracies, marker='o', label=label)
    plt.xlabel('Board Position')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Across Board Positions for Each Probe Point (Linear Probing)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('assets/txt_accuracy_across_positions_all_points.png')
    plt.close()

    # New: Heatmap of accuracies
    accuracies = np.array([[result['val_accuracy'] for result in results] for results in all_results.values()])
    plt.figure(figsize=(15, 10))
    plt.imshow(accuracies, cmap='viridis', aspect='auto')
    plt.colorbar(label='Validation Accuracy')
    plt.xlabel('Board Position')
    plt.ylabel('Probe Point')
    plt.title('Heatmap of Accuracies Across Probe Points and Board Positions')
    plt.yticks(range(len(probe_points)), x_labels)
    plt.xticks(range(9), positions)
    plt.tight_layout()
    plt.savefig('assets/txt_accuracy_heatmap.png')
    plt.close()

    print("Graphs have been saved in the assets directory.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("out-txt-models/ckpt_iter_5000.pt").to(device)

    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])

    # Load and sample games
    with open("data/txt/all_tic_tac_toe_games.csv", 'r') as file:
        all_games = [f";{row.split(',')[0]}" for row in file.readlines()[1:]]

    num_games_to_select = len(all_games) // 20
    games = random.sample(all_games, num_games_to_select)

    print(f"Processing {len(games)} games (5% of total)")

    labels = prepare_labels(games)

    all_results = process_all_points(model, games, tokenizer, device, labels, max_iter=1000)

    if all_results:
        print("Probing results for all points saved.")
        generate_graphs(all_results)
    else:
        print("Error: No results were generated during probing.")

if __name__ == "__main__":
    main()