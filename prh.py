import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
from extract_features import load_model, tokenize_and_pad
from img_game_generator import create_board_image
from data.img.prepare_img import load_and_process_image
from alignment_metrics import AlignmentMetrics

def process_move(move):
    if len(move) != 3:
        return None
    try:
        row = int(move[1])
        col = int(move[2])
        if 1 <= row <= 3 and 1 <= col <= 3:
            return move
    except ValueError:
        pass
    return None

def extract_activations(model, input_tensor):
    activations = {}
    
    # Embedding layer
    tok_emb = model.transformer.wte(input_tensor)
    pos_emb = model.transformer.wpe(torch.arange(input_tensor.size(1), device=input_tensor.device))
    x = model.transformer.drop(tok_emb + pos_emb)
    activations['embedding'] = x.detach().cpu().numpy()
    
    # Transformer layers
    for i, block in enumerate(model.transformer.h):
        ln1_out = block.ln_1(x)
        attn_output = block.attn(ln1_out)
        x = x + attn_output
        activations[f'layer_{i}_attn'] = x.detach().cpu().numpy()
        
        ln2_out = block.ln_2(x)
        mlp_output = block.mlp(ln2_out)
        x = x + mlp_output
        activations[f'layer_{i}_mlp'] = x.detach().cpu().numpy()
    
    # Final layer norm
    x = model.transformer.ln_f(x)
    activations['final_ln'] = x.detach().cpu().numpy()
    
    # Language model head
    logits = model.lm_head(x)
    activations['lm_head'] = logits.detach().cpu().numpy()
    
    return activations

def extract_representations(model, games, is_image_model, device, tokenizer=None):
    model.eval()
    all_activations = []
    
    for i, game in enumerate(games):
        with torch.no_grad():
            if is_image_model:
                print(f"Processing game {i}: {game}")
                moves = game[1:].split()  # Supprimer le point-virgule initial
                valid_moves = [process_move(move) for move in moves if process_move(move) is not None]
                
                if len(valid_moves) != len(moves):
                    print(f"Warning: Game {i} has invalid moves. Original: {moves}, Processed: {valid_moves}")
                    continue  # Passer à la partie suivante si des mouvements sont invalides
                
                try:
                    img = create_board_image(valid_moves)
                    input_data = load_and_process_image(img)
                    input_data = np.insert(input_data, 0, tokenizer(';'))
                    input_tensor = torch.from_numpy(np.array([input_data])).to(device).long()
                    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
                except Exception as e:
                    print(f"Error processing game {i}: {str(e)}")
                    continue
            else:
                try:
                    if not game.startswith(';'):
                        game = ';' + game
                    input_tensor = tokenize_and_pad([game], tokenizer, model.config.block_size).to(device)
                except Exception as e:
                    print(f"Error processing text game {i}: {str(e)}")
                    continue
            
            activations = extract_activations(model, input_tensor)
            all_activations.append(activations)
    
    if not all_activations:
        raise ValueError("No valid activations were generated.")
    
    return all_activations

def compare_activations(img_activations, txt_activations):
    results = {}
    
    for layer in img_activations[0].keys():
        img_layer_activations = np.array([act[layer] for act in img_activations])
        txt_layer_activations = np.array([act[layer] for act in txt_activations])
        
        # Ensure the activations have the same shape
        min_games = min(img_layer_activations.shape[0], txt_layer_activations.shape[0])
        img_layer_activations = img_layer_activations[:min_games]
        txt_layer_activations = txt_layer_activations[:min_games]
        
        # Flatten the activations
        img_flat = img_layer_activations.reshape(min_games, -1)
        txt_flat = txt_layer_activations.reshape(min_games, -1)
        
        metrics = ['cycle_knn', 'mutual_knn', 'cka', 'svcca']
        layer_results = {}
        
        for metric in metrics:
            if metric in ['cycle_knn', 'mutual_knn']:
                result = AlignmentMetrics.measure(metric, torch.tensor(img_flat), torch.tensor(txt_flat), topk=5)
            elif metric == 'cka':
                result = AlignmentMetrics.measure(metric, torch.tensor(img_flat), torch.tensor(txt_flat))
            elif metric == 'svcca':
                # Calculer le nombre maximum de composants pour SVCCA
                max_components = min(img_flat.shape[0], img_flat.shape[1], txt_flat.shape[1])
                # Limiter le nombre de composants à une valeur raisonnable (par exemple, 100)
                n_components = min(64, max_components)
                result = AlignmentMetrics.measure(metric, torch.tensor(img_flat), torch.tensor(txt_flat), cca_dim=n_components)
            
            layer_results[metric] = result
        
        results[layer] = layer_results
    
    return results

def generate_graph(results):
    metrics = list(next(iter(results.values())).keys())
    layers = list(results.keys())
    
    plt.figure(figsize=(12, 6))
    
    for metric in metrics:
        values = [results[layer][metric] for layer in layers]
        plt.plot(values, marker='o', label=metric)
    
    plt.xlabel('Layers')
    plt.ylabel('Metric Value')
    plt.title('Comparison of Metrics Across Layers')
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    print("Graph saved as 'metrics_comparison.png'")

def generate_csv(results):
    with open('assets/prh_metrics_comparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        metrics = list(next(iter(results.values())).keys())
        writer.writerow(['Layer'] + metrics)
        
        for layer, layer_results in results.items():
            writer.writerow([layer] + [layer_results[metric] for metric in metrics])
    
    print("CSV file saved as 'metrics_comparison.csv'")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_model = load_model("out-img-models/ckpt_iter_3000.pt").to(device)
    txt_model = load_model("out-img-models/ckpt_iter_5000.pt").to(device)
    
    with open("data/txt/all_tic_tac_toe_games.csv", 'r') as file:
        all_games = [f";{row.split(',')[0]}" for row in file]
    
    games = all_games[:100]  # Échantillon de 100 jeux pour le test
    
    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])
    
    try:
        img_activations = extract_representations(img_model, games, True, device, tokenizer)
        txt_activations = extract_representations(txt_model, games, False, device, tokenizer)
        
        results = compare_activations(img_activations, txt_activations)
        
        print("Résultats de la comparaison des modèles:")
        for layer, layer_results in results.items():
            print(f"\nLayer: {layer}")
            for metric, value in layer_results.items():
                print(f"  {metric}: {value}")
        
        generate_graph(results)
        generate_csv(results)
        
    except Exception as e:
        print(f"Une erreur est survenue lors de la comparaison des modèles: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()