import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
import random
import csv
from typing import List, Tuple, Dict

# Assuming these functions are defined in extract_features.py
from extract_features import load_model, tokenize_and_pad
from txt_game_generator import is_winner

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

def get_probe_points(model: nn.Module) -> List[str]:
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

def extract_activations_all_points(model: nn.Module, games: List[str], tokenizer, device: torch.device, batch_size: int = 4, chunk_size: int = 100) -> Dict[str, np.ndarray]:
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
                embedding = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(1), device=device))
                x = model.transformer.drop(embedding)

                all_activations['embedding'].append(embedding.cpu())

                for layer_idx, block in enumerate(model.transformer.h):
                    ln1_out = block.ln_1(x)
                    attn_output = block.attn(ln1_out)
                    x = x + attn_output

                    ln2_out = block.ln_2(x)
                    mlp_output = block.mlp(ln2_out)
                    x = x + mlp_output

                    all_activations[f'layer{layer_idx}_pre_attn_norm'].append(ln1_out.cpu())
                    all_activations[f'layer{layer_idx}_attn'].append(attn_output.cpu())
                    all_activations[f'layer{layer_idx}_post_attn'].append(x.cpu())
                    all_activations[f'layer{layer_idx}_pre_ffn_norm'].append(ln2_out.cpu())
                    all_activations[f'layer{layer_idx}_mlp_fc'].append(block.mlp.c_fc(ln2_out).cpu())
                    all_activations[f'layer{layer_idx}_mlp_proj'].append(mlp_output.cpu())

                x = model.transformer.ln_f(x)
                all_activations['final_ln'].append(x.cpu())
                all_activations['lm_head'].append(model.lm_head(x).cpu())

    for point, activations in all_activations.items():
        if activations:
            all_activations[point] = torch.cat(activations, dim=0).cpu().numpy()
        else:
            print(f"Warning: No activations for {point} across all chunks")
            all_activations[point] = np.array([])

    return all_activations

def train_sae(activations: np.ndarray, hidden_dim: int = 100, learning_rate: float = 1e-3, num_epochs: int = 100, l1_lambda: float = 1e-5) -> Tuple[SparseAutoencoder, List[float], List[float]]:
    activations_2d = activations.reshape(-1, activations.shape[-1])
    
    input_dim = activations_2d.shape[1]
    sae = SparseAutoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    X_train, X_val = train_test_split(activations_2d, test_size=0.2, random_state=42)
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)

    train_losses = []
    val_losses = []

    for _ in tqdm(range(num_epochs), desc="Training SAE"):
        sae.train()
        optimizer.zero_grad()
        _, decoded = sae(X_train)
        loss = criterion(decoded, X_train)
        l1_loss = sum(p.abs().sum() for p in sae.encoder.parameters())
        total_loss = loss + l1_lambda * l1_loss
        total_loss.backward()
        optimizer.step()
        train_losses.append(total_loss.item())

        sae.eval()
        with torch.no_grad():
            _, val_decoded = sae(X_val)
            val_loss = criterion(val_decoded, X_val)
            val_losses.append(val_loss.item())

    return sae, train_losses, val_losses

def is_perfect_game(game: str) -> bool:
    moves = game.split()
    board = [[' ' for _ in range(3)] for _ in range(3)]
    
    for i, move in enumerate(moves):
        player = move[0]
        row = int(move[1]) - 1
        col = int(move[2]) - 1
        board[row][col] = player
        
        if is_winner(board, player):
            return False
    
    return len(moves) == 9

def create_balanced_dataset(all_games: List[Tuple[str, str]], num_samples: int) -> List[str]:
    perfect_games = []
    non_perfect_games = []
    
    for game, result in all_games:
        if result == '1/2-1/2':
            perfect_games.append(game)
        else:
            non_perfect_games.append(game)
    
    num_each = num_samples // 2
    if len(perfect_games) < num_each or len(non_perfect_games) < num_each:
        raise ValueError(f"Not enough games of each type. Perfect: {len(perfect_games)}, Non-perfect: {len(non_perfect_games)}")
    
    balanced_games = random.sample(perfect_games, num_each) + random.sample(non_perfect_games, num_each)
    random.shuffle(balanced_games)
    
    return balanced_games

def analyze_sae_neurons_with_perfection(sae: SparseAutoencoder, activations: np.ndarray, labels: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    sae.eval()
    activations_2d = activations.reshape(-1, activations.shape[-1])
    with torch.no_grad():
        encoded, _ = sae(torch.FloatTensor(activations_2d))
    
    neuron_activations = encoded.numpy()
    
    # Reshape neuron_activations to match labels shape
    neuron_activations = neuron_activations.reshape(labels.shape[0], -1, neuron_activations.shape[-1])
    neuron_activations = neuron_activations.mean(axis=1)  # Average over sequence length
    
    perfect_mask = labels == 1
    perfect_activations = neuron_activations[perfect_mask]
    non_perfect_activations = neuron_activations[~perfect_mask]
    
    mean_diff = perfect_activations.mean(axis=0) - non_perfect_activations.mean(axis=0)
    
    top_neurons = np.argsort(np.abs(mean_diff))[-top_k:]
    
    return top_neurons, mean_diff

def process_all_points_with_sae(model: nn.Module, games: List[str], tokenizer, device: torch.device, hidden_dim: int = 100) -> Dict[str, Dict]:
    try:
        print("Starting to process all probe points with SAE")
        all_activations = extract_activations_all_points(model, games, tokenizer, device)
        
        labels = np.array([is_perfect_game(game) for game in games])

        all_results = {}
        for point, activations in all_activations.items():
            if activations.size == 0:
                print(f"Skipping {point} due to empty activations")
                continue

            print(f"Processing probe point: {point}")
            print(f"Activations shape: {activations.shape}")

            sae, train_losses, val_losses = train_sae(activations, hidden_dim=hidden_dim)
            top_neurons, mean_diff = analyze_sae_neurons_with_perfection(sae, activations, labels)

            results = {
                'sae': sae,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'top_neurons': top_neurons,
                'mean_diff': mean_diff
            }

            all_results[point] = results

            with open(f"sae_results_{point}.pkl", 'wb') as f:
                pickle.dump(results, f)

        return all_results
    except Exception as e:
        print(f"Error processing probe points with SAE: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_sae_graphs(all_results: Dict[str, Dict]):
    os.makedirs('assets', exist_ok=True)
    
    for point, results in all_results.items():
        plt.figure(figsize=(10, 5))
        plt.plot(results['train_losses'], label='Train Loss')
        plt.plot(results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'SAE Learning Curves - {point}')
        plt.legend()
        plt.savefig(f'assets/sae_learning_curves_{point}.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=range(len(results['mean_diff'])), y=results['mean_diff'])
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Difference (Perfect - Non-Perfect)')
        plt.title(f'Neuron Activation Differences for Perfect vs Non-Perfect Games - {point}')
        plt.savefig(f'assets/sae_perfect_play_diff_{point}.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        top_diffs = results['mean_diff'][results['top_neurons']]
        plt.bar(range(len(top_diffs)), top_diffs)
        plt.xlabel('Top Neuron Index')
        plt.ylabel('Mean Activation Difference')
        plt.title(f'Top {len(results["top_neurons"])} Neurons for Perfect Play - {point}')
        plt.savefig(f'assets/sae_top_neurons_perfect_play_{point}.png')
        plt.close()

    print("SAE graphs have been saved in the assets directory.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("out-txt-models/ckpt_iter_3200.pt").to(device)

    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])

    with open("data/txt/all_tic_tac_toe_games.csv", 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        all_games = [(row[0], row[1]) for row in csv_reader]

    num_samples = 5000  # Total number of games to use
    try:
        games = create_balanced_dataset(all_games, num_samples)
    except ValueError as e:
        print(f"Error creating balanced dataset: {e}")
        return

    print(f"Created balanced dataset with {len(games)} games")

    all_results = process_all_points_with_sae(model, games, tokenizer, device)
    
    if all_results:
        generate_sae_graphs(all_results)

        with open('sae_results_perfect_play.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Probe Point', 'Final Train Loss', 'Final Val Loss', 'Top Neurons for Perfect Play', 'Mean Activation Difference'])
            for point, results in all_results.items():
                writer.writerow([
                    point,
                    results['train_losses'][-1],
                    results['val_losses'][-1],
                    ', '.join(map(str, results['top_neurons'])),
                    ', '.join(map(str, results['mean_diff'][results['top_neurons']]))
                ])

if __name__ == "__main__":
    main()