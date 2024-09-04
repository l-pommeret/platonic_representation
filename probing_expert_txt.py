import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import os
from extract_features import load_model, tokenize_and_pad

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded


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

        for i in range(0, len(chunk_games), batch_size):
            batch = chunk_games[i:i+batch_size]
            input_ids = tokenize_and_pad(batch, tokenizer, max_length).to(device)

            with torch.no_grad():
                # Embedding
                embedding = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(1), device=device))
                x = model.transformer.drop(embedding)

                all_activations['embedding'].append(embedding.cpu())

                # Process all layers
                for layer_idx, block in enumerate(model.transformer.h):
                    ln1_out = block.ln_1(x)
                    attn_output = block.attn(ln1_out)
                    x = x + attn_output

                    ln2_out = block.ln_2(x)
                    mlp_output = block.mlp(ln2_out)
                    x = x + mlp_output

                    # Store activations
                    all_activations[f'layer{layer_idx}_pre_attn_norm'].append(ln1_out.cpu())
                    all_activations[f'layer{layer_idx}_attn'].append(attn_output.cpu())
                    all_activations[f'layer{layer_idx}_post_attn'].append(x.cpu())
                    all_activations[f'layer{layer_idx}_pre_ffn_norm'].append(ln2_out.cpu())
                    all_activations[f'layer{layer_idx}_mlp_fc'].append(block.mlp.c_fc(ln2_out).cpu())
                    all_activations[f'layer{layer_idx}_mlp_proj'].append(mlp_output.cpu())

                # Final layer norm and language model head
                x = model.transformer.ln_f(x)
                all_activations['final_ln'].append(x.cpu())
                all_activations['lm_head'].append(model.lm_head(x).cpu())

    # Concatenate all chunks
    for point, activations in all_activations.items():
        if activations:
            all_activations[point] = torch.cat(activations, dim=0).numpy()
        else:
            print(f"Warning: No activations for {point} across all chunks")
            all_activations[point] = np.array([])

    return all_activations

def prepare_labels(games):
    """Prepare labels for win/draw classification."""
    labels = []
    for game in games:
        result = game.split(',')[1].strip()
        if result == '1/2-1/2':
            labels.append(1)  # Draw
        else:
            labels.append(0)  # Win (for either X or O)
    return np.array(labels)

def prepare_balanced_dataset(games, labels, n_samples=1000):
    """Prepare a balanced dataset with equal number of wins and draws."""
    win_indices = np.where(labels == 0)[0]
    draw_indices = np.where(labels == 1)[0]
    
    n_samples_per_class = min(len(win_indices), len(draw_indices), n_samples // 2)
    
    balanced_win_indices = np.random.choice(win_indices, n_samples_per_class, replace=False)
    balanced_draw_indices = np.random.choice(draw_indices, n_samples_per_class, replace=False)
    
    balanced_indices = np.concatenate([balanced_win_indices, balanced_draw_indices])
    np.random.shuffle(balanced_indices)
    
    return [games[i] for i in balanced_indices], labels[balanced_indices]

def train_and_evaluate_sae(activations, labels, encoding_dim, device, n_splits=5, epochs=100, batch_size=32):
    """Train Sparse Autoencoder and evaluate using Logistic Regression."""
    results = []
    
    if activations.ndim == 3:
        activations = activations.mean(axis=1)

    scaler = StandardScaler()
    activations_normalized = scaler.fit_transform(activations)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(activations_normalized):
        X_train, X_val = activations_normalized[train_index], activations_normalized[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        sae = SparseAutoencoder(X_train.shape[1], encoding_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(sae.parameters())

        # Train SAE
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
                optimizer.zero_grad()
                encoded, decoded = sae(batch)
                loss = criterion(decoded, batch) + 0.0001 * torch.sum(torch.abs(encoded))  # L1 regularization
                loss.backward()
                optimizer.step()

        # Evaluate using Logistic Regression
        sae.eval()
        with torch.no_grad():
            train_encoded, _ = sae(torch.FloatTensor(X_train).to(device))
            val_encoded, _ = sae(torch.FloatTensor(X_val).to(device))

        train_encoded = train_encoded.cpu().numpy()
        val_encoded = val_encoded.cpu().numpy()

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(train_encoded, y_train)

        train_pred = clf.predict(train_encoded)
        val_pred = clf.predict(val_encoded)

        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)

        results.append({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        })

    return results

def process_all_points(model, games, tokenizer, device, labels, encoding_dim=64):
    """Process all probe points: extract activations and train SAE for probing."""
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

            results = train_and_evaluate_sae(activations, labels, encoding_dim, device)

            avg_train_accuracy = np.mean([result['train_accuracy'] for result in results])
            avg_val_accuracy = np.mean([result['val_accuracy'] for result in results])

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
    plt.title('Average Accuracy Across Probe Points (SAE Probing)')
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
    plt.ylim(0.45, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_across_probe_points_sae.png')
    plt.close()

    # Generate CSV file
    with open('validation_accuracy_sae.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Probe Point', 'Average Validation Accuracy'])
        for point, results in all_results.items():
            avg_accuracy = np.mean([result['val_accuracy'] for result in results])
            writer.writerow([point, avg_accuracy])

    print("CSV file 'validation_accuracy_sae.csv' has been generated.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("out-txt-models/ckpt_iter_3200_rand.pt").to(device)

    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])

    # Load and prepare balanced dataset
    with open("data/txt/all_tic_tac_toe_games.csv", 'r') as file:
        all_games = [f";{row}" for row in file]

    labels = prepare_labels(all_games)
    games, labels = prepare_balanced_dataset(all_games, labels, n_samples=2000)

    all_results = process_all_points(model, games, tokenizer, device, labels)
    
    if all_results:
        generate_graphs(all_results)

if __name__ == "__main__":
    main()