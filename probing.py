import torch
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle
from tqdm import tqdm
import os

from extract_features import load_model, tokenize_and_pad

def tokenize_and_pad(texts, tokenizer, max_length):
    tokenized = [[tokenizer(c) for c in text] for text in texts]
    padded = torch.full((len(texts), max_length), tokenizer(';'))  # Padding token
    for i, seq in enumerate(tokenized):
        length = min(len(seq), max_length)
        padded[i, :length] = torch.tensor(seq[:length])
    return padded

def extract_activations_single_layer(model, games, tokenizer, device, layer_num, batch_size=8, chunk_size=500):
    model.eval()
    max_length = model.config.block_size
    all_activations = []
    
    for chunk_start in tqdm(range(0, len(games), chunk_size), desc=f"Processing layer {layer_num}"):
        chunk_end = min(chunk_start + chunk_size, len(games))
        chunk_games = games[chunk_start:chunk_end]
        
        chunk_activations = []
        
        for i in range(0, len(chunk_games), batch_size):
            batch = chunk_games[i:i+batch_size]
            input_ids = tokenize_and_pad(batch, tokenizer, max_length)
            input_ids = input_ids.to(device)
            
            with torch.no_grad():
                _, _, hidden_states = model(input_ids)
            
            activations = hidden_states[layer_num].cpu()
            chunk_activations.append(activations)
            
            del hidden_states
            torch.cuda.empty_cache()
        
        chunk_activations = torch.cat(chunk_activations, dim=0)
        all_activations.append(chunk_activations.to('cpu'))
        del chunk_activations
        torch.cuda.empty_cache()
    
    return torch.cat(all_activations, dim=0)

def prepare_labels(games):
    labels = []
    for game in games:
        board = ['-'] * 9
        moves = game[1:].split()  # Remove the leading ';' and split moves
        for i, move in enumerate(moves):
            player = 'X' if i % 2 == 0 else 'O'
            position = int(move[1]) - 1 + 3 * (int(move[2]) - 1)
            board[position] = player
        labels.append(board)
    return np.array(labels)

def train_and_evaluate_probing_classifiers(activations, labels):
    results = []
    n_splits = 5
    
    for position in range(9):
        y = labels[:, position]
        
        clf = LogisticRegression(multi_class='ovr', max_iter=1000)
        
        # Perform cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []
        
        for train_index, val_index in kf.split(activations):
            X_train, X_val = activations[train_index], activations[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            clf.fit(X_train, y_train)
            
            # Training metrics
            train_pred = clf.predict(X_train)
            train_prob = clf.predict_proba(X_train)
            train_acc = (train_pred == y_train).mean()
            train_loss = log_loss(y_train, train_prob)
            
            # Validation metrics
            val_pred = clf.predict(X_val)
            val_prob = clf.predict_proba(X_val)
            val_acc = (val_pred == y_val).mean()
            val_loss = log_loss(y_val, val_prob)
            
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
        
        # Average metrics across folds
        results.append({
            'position': position,
            'train_accuracy': np.mean(train_accuracies),
            'train_loss': np.mean(train_losses),
            'val_accuracy': np.mean(val_accuracies),
            'val_loss': np.mean(val_losses)
        })
    
    return results

def save_activations(activations, layer):
    # Convert to numpy and save
    np_activations = activations.numpy()
    
    # Create directory if it doesn't exist
    os.makedirs('activations', exist_ok=True)
    
    # Save in chunks of 1000 samples
    chunk_size = 1000
    for i in range(0, len(np_activations), chunk_size):
        chunk = np_activations[i:i+chunk_size]
        np.save(f'activations/activations_layer_{layer}_chunk_{i//chunk_size}.npy', chunk)

def main():
    model = load_model("out-txt-models/ckpt_iter_5000.pt")
    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])
    
    with open("data/txt/all_tic_tac_toe_games.csv", 'r') as file:
        games = [f";{row.split(',')[0]}" for row in file.readlines()[1:]]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    labels = prepare_labels(games)
    
    for layer in range(model.config.n_layer + 2):  # +2 for input embedding and final layer
        print(f"Processing layer {layer}")
        activations = extract_activations_single_layer(model, games, tokenizer, device, layer)
        
        # Save activations to disk using numpy
        save_activations(activations.cpu(), layer)
        
        results = train_and_evaluate_probing_classifiers(activations, labels)
        
        # Print and save results for this layer
        print(f"Layer {layer}:")
        for pos_result in results:
            print(f"  Position {pos_result['position'] + 1}:")
            print(f"    Train Accuracy: {pos_result['train_accuracy']:.4f}")
            print(f"    Train Loss: {pos_result['train_loss']:.4f}")
            print(f"    Validation Accuracy: {pos_result['val_accuracy']:.4f}")
            print(f"    Validation Loss: {pos_result['val_loss']:.4f}")
        print()
        
        # Save results for this layer
        os.makedirs('results', exist_ok=True)
        with open(f"results/probing_results_layer_{layer}.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Clear memory
        del activations
        torch.cuda.empty_cache()
    
    print("Probing results for all layers saved.")

if __name__ == "__main__":
    main()