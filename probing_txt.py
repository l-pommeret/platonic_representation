import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
import pickle
from tqdm import tqdm
import os
import traceback
import random
import matplotlib.pyplot as plt

from extract_features import load_model

def tokenize_and_pad(texts, tokenizer, max_length):
    """Tokenize and pad a list of texts."""
    tokenized = [[tokenizer(c) for c in text] for text in texts]
    padded = torch.full((len(texts), max_length), tokenizer(';'))
    for i, seq in enumerate(tokenized):
        length = min(len(seq), max_length)
        padded[i, :length] = torch.tensor(seq[:length])
    return padded

def extract_activations_all_points(model, games, tokenizer, device, batch_size=4, chunk_size=100):
    """Extract activations for all possible points in the model."""
    model.eval()
    max_length = model.config.block_size
    
    probe_points = [
        'embedding',
        'layer0_pre_attn_norm', 'layer0_attn', 'layer0_pre_ffn_norm', 'layer0_mlp',
        'layer1_pre_attn_norm', 'layer1_attn', 'layer1_pre_ffn_norm', 'layer1_mlp',
        'final_ln',
        'lm_head'
    ]
    
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
                
                # Layer 0
                layer0 = model.transformer.h[0]
                ln1_out = layer0.ln_1(x)
                chunk_activations['layer0_pre_attn_norm'].append(ln1_out.cpu())
                
                attn_output = layer0.attn(ln1_out)
                chunk_activations['layer0_attn'].append(attn_output.cpu())
                
                x = x + attn_output
                ln2_out = layer0.ln_2(x)
                chunk_activations['layer0_pre_ffn_norm'].append(ln2_out.cpu())
                
                mlp_output = layer0.mlp(ln2_out)
                chunk_activations['layer0_mlp'].append(mlp_output.cpu())
                
                x = x + mlp_output
                
                # Layer 1
                layer1 = model.transformer.h[1]
                ln1_out = layer1.ln_1(x)
                chunk_activations['layer1_pre_attn_norm'].append(ln1_out.cpu())
                
                attn_output = layer1.attn(ln1_out)
                chunk_activations['layer1_attn'].append(attn_output.cpu())
                
                x = x + attn_output
                ln2_out = layer1.ln_2(x)
                chunk_activations['layer1_pre_ffn_norm'].append(ln2_out.cpu())
                
                mlp_output = layer1.mlp(ln2_out)
                chunk_activations['layer1_mlp'].append(mlp_output.cpu())
                
                x = x + mlp_output
                
                # Final layer norm
                x = model.transformer.ln_f(x)
                chunk_activations['final_ln'].append(x.cpu())
                
                # Language model head
                lm_output = model.lm_head(x)
                chunk_activations['lm_head'].append(lm_output.cpu())
        
        # Concatenate and store chunk activations
        for point in probe_points:
            all_activations[point].append(torch.cat(chunk_activations[point], dim=0))
        
        del chunk_activations
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    for point in probe_points:
        all_activations[point] = torch.cat(all_activations[point], dim=0).numpy()
    
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

def train_and_evaluate_probing_classifiers(activations, labels):
    """Train and evaluate probing classifiers for each board position."""
    results = []
    n_splits = 5
    
    if activations.ndim == 3:
        activations = activations.mean(axis=1)
    
    for position in range(9):
        y = labels[:, position]
        base_clf = LogisticRegression(max_iter=10000)
        clf = OneVsRestClassifier(base_clf)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_metrics = []
        val_metrics = []
        
        for train_index, val_index in kf.split(activations):
            X_train, X_val = activations[train_index], activations[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            clf.fit(X_train, y_train)
            
            train_pred = clf.predict(X_train)
            train_prob = clf.predict_proba(X_train)
            val_pred = clf.predict(X_val)
            val_prob = clf.predict_proba(X_val)
            
            train_metrics.append({
                'accuracy': (train_pred == y_train).mean(),
                'loss': log_loss(y_train, train_prob)
            })
            val_metrics.append({
                'accuracy': (val_pred == y_val).mean(),
                'loss': log_loss(y_val, val_prob)
            })
        
        results.append({
            'position': position,
            'train_accuracy': np.mean([m['accuracy'] for m in train_metrics]),
            'train_loss': np.mean([m['loss'] for m in train_metrics]),
            'val_accuracy': np.mean([m['accuracy'] for m in val_metrics]),
            'val_loss': np.mean([m['loss'] for m in val_metrics])
        })
    
    return results

def process_all_points(model, games, tokenizer, device, labels):
    """Process all probe points: extract activations and train probing classifiers."""
    try:
        print("Starting to process all probe points")
        all_activations = extract_activations_all_points(model, games, tokenizer, device)
        
        all_results = {}
        for point, activations in all_activations.items():
            print(f"Processing probe point: {point}")
            print(f"Activations shape: {activations.shape}")
            
            results = train_and_evaluate_probing_classifiers(activations, labels)
            
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
        print("Traceback:")
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
            layer, component = point.split('_')
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

    print("Graphs have been saved in the assets directory.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("out-txt-models/ckpt_iter_0.pt").to(device)
    
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
    
    all_results = process_all_points(model, games, tokenizer, device, labels)
    
    print("Probing results for all points saved.")

    generate_graphs(all_results)

if __name__ == "__main__":
    main()