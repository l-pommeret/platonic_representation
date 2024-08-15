import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle
from tqdm import tqdm
import os
import traceback
import random

import matplotlib.pyplot as plt

from extract_features import load_model

def tokenize_and_pad(texts, tokenizer, max_length):
    tokenized = [[tokenizer(c) for c in text] for text in texts]
    padded = torch.full((len(texts), max_length), tokenizer(';'))
    for i, seq in enumerate(tokenized):
        length = min(len(seq), max_length)
        padded[i, :length] = torch.tensor(seq[:length])
    return padded

def extract_activations_single_layer(model, games, tokenizer, device, layer_num, batch_size=4, chunk_size=100):
    model.eval()
    max_length = model.config.block_size
    
    sample_input = torch.tensor([[tokenizer(';')] * max_length], device=device)
    with torch.no_grad():
        _, _, sample_hidden_states = model(sample_input)
        sample_activation = sample_hidden_states[layer_num]
    
    activation_shape = (len(games),) + sample_activation.shape[1:]
    memmap_activations = np.memmap('temp_activations.dat', dtype='float32', mode='w+', shape=activation_shape)
    
    for chunk_start in tqdm(range(0, len(games), chunk_size), desc=f"Processing layer {layer_num}"):
        chunk_end = min(chunk_start + chunk_size, len(games))
        chunk_games = games[chunk_start:chunk_end]
        
        chunk_activations = []
        
        for i in range(0, len(chunk_games), batch_size):
            batch = chunk_games[i:i+batch_size]
            input_ids = tokenize_and_pad(batch, tokenizer, max_length).to(device)
            
            with torch.no_grad():
                _, _, hidden_states = model(input_ids)
            
            activations = hidden_states[layer_num].cpu().numpy()
            chunk_activations.append(activations)
        
        chunk_activations = np.concatenate(chunk_activations, axis=0)
        memmap_activations[chunk_start:chunk_end] = chunk_activations
        
        del chunk_activations
        torch.cuda.empty_cache()
    
    return memmap_activations

def prepare_labels(games):
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

from sklearn.multiclass import OneVsRestClassifier

def train_and_evaluate_probing_classifiers(activations, labels):
    results = []
    n_splits = 5
    
    if activations.ndim == 3:
        activations = activations.mean(axis=1)
    
    for position in range(9):
        y = labels[:, position]
        base_clf = LogisticRegression(max_iter=5000)
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

def process_layer(model, games, tokenizer, device, layer, labels):
    try:
        print(f"Starting to process layer {layer}")
        activations = extract_activations_single_layer(model, games, tokenizer, device, layer)
        print(f"Activations shape after extraction: {activations.shape}")
        
        results = train_and_evaluate_probing_classifiers(activations, labels)
        
        print(f"Layer {layer}:")
        avg_train_accuracy = 0
        avg_val_accuracy = 0
        for pos_result in results:
            print(f"  Position {pos_result['position'] + 1}:")
            print(f"    Train Accuracy: {pos_result['train_accuracy']:.4f}")
            print(f"    Train Loss: {pos_result['train_loss']:.4f}")
            print(f"    Validation Accuracy: {pos_result['val_accuracy']:.4f}")
            print(f"    Validation Loss: {pos_result['val_loss']:.4f}")
            avg_train_accuracy += pos_result['train_accuracy']
            avg_val_accuracy += pos_result['val_accuracy']
        
        avg_train_accuracy /= 9
        avg_val_accuracy /= 9
        print(f"\nLayer {layer} Summary:")
        print(f"  Average Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"  Average Validation Accuracy: {avg_val_accuracy:.4f}")
        print()
        
        with open(f"probing_results_layer_{layer}.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        del activations
        torch.cuda.empty_cache()
        
        if os.path.exists('temp_activations.dat'):
            os.remove('temp_activations.dat')
        
        return results
    except Exception as e:
        print(f"Error processing layer {layer}: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return None

def generate_graphs(all_results):
    layers = list(range(len(all_results)))
    positions = list(range(1, 10))
    accuracies = [[result['val_accuracy'] for result in layer_results] for layer_results in all_results]

    # Plotting average accuracy across layers
    plt.figure(figsize=(10, 6))
    avg_accuracies = [np.mean(acc) for acc in accuracies]
    plt.plot(layers, avg_accuracies, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Average Accuracy Across Layers (Linear Probing)')
    plt.xticks(layers)  # Ensure only whole number layers are shown
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assets/accuracy_across_layers.png')
    plt.close()

    # Plotting accuracy across positions
    plt.figure(figsize=(10, 6))
    for i, layer in enumerate(layers):
        plt.plot(positions, accuracies[i], marker='o', label=f'Layer {layer}')
    plt.xlabel('Position')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Across Positions for Each Layer (Linear Probing)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('assets/accuracy_across_positions.png')
    plt.close()

    print("Graphs have been saved in the assets directory.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("out-txt-models/ckpt_iter_10000.pt").to(device)
    
    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])
    
    # Load only 10% of the data
    with open("data/txt/all_tic_tac_toe_games.csv", 'r') as file:
        all_games = [f";{row.split(',')[0]}" for row in file.readlines()[1:]]
    
    # Randomly select 20% of the games
    num_games_to_select = len(all_games) // 20
    games = random.sample(all_games, num_games_to_select)
    
    print(f"Processing {len(games)} games (20% of total)")
    
    labels = prepare_labels(games)
    
    all_results = []
    for layer in range(model.config.n_layer + 2):
        print(f"Processing layer {layer}")
        layer_results = process_layer(model, games, tokenizer, device, layer, labels)
        if layer_results:
            all_results.append(layer_results)
    
    print("Probing results for all layers saved.")

    # Generate graphs
    generate_graphs(all_results)

if __name__ == "__main__":
    main()