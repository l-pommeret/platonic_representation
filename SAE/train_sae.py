import torch
import pickle
import os
import random
from model import GPTConfig, GPT

# Chargement des données d'encodage
with open('data/txt/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']

# Fonctions d'encodage et de décodage
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])
encode_game = lambda game: encode(' '.join(game))

class TicTacToe:
    UNKNOWN, USER, COMPUTER = range(3)
    DIAGONAL_A = [(0, 0), (1, 1), (2, 2)]
    DIAGONAL_B = [(0, 2), (1, 1), (2, 0)]

    def __init__(self):
        self.board = [[self.UNKNOWN for _ in range(3)] for _ in range(3)]
        self.count = 0
        self.row_sum = [0, 0, 0]
        self.col_sum = [0, 0, 0]
        self.diag_a_sum = 0
        self.diag_b_sum = 0

    def put_mark(self, player, coordinate):
        self.count += 1
        self.board[coordinate[0]][coordinate[1]] = player
        score = 1 if player == self.USER else 4
        self.row_sum[coordinate[0]] += score
        self.col_sum[coordinate[1]] += score
        if coordinate in self.DIAGONAL_A:
            self.diag_a_sum += score
        if coordinate in self.DIAGONAL_B:
            self.diag_b_sum += score

    def check_winner(self):
        for i in range(3):
            if self.row_sum[i] == 3 or self.col_sum[i] == 3:
                return self.USER
            if self.row_sum[i] == 12 or self.col_sum[i] == 12:
                return self.COMPUTER
        if self.diag_a_sum in (3, 12) or self.diag_b_sum in (3, 12):
            return self.USER if self.diag_a_sum == 3 or self.diag_b_sum == 3 else self.COMPUTER
        return self.UNKNOWN

    def is_full(self):
        return self.count == 9

    def get_empty_cells(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == self.UNKNOWN]

    def potential_win_check(self, player):
        test = 1 if player == self.USER else 4
        for i in range(3):
            if self.row_sum[i] == 2 * test:
                return next((i, j) for j in range(3) if self.board[i][j] == self.UNKNOWN)
            if self.col_sum[i] == 2 * test:
                return next((j, i) for j in range(3) if self.board[j][i] == self.UNKNOWN)
        if self.diag_a_sum == 2 * test:
            return next(pos for pos in self.DIAGONAL_A if self.board[pos[0]][pos[1]] == self.UNKNOWN)
        if self.diag_b_sum == 2 * test:
            return next(pos for pos in self.DIAGONAL_B if self.board[pos[0]][pos[1]] == self.UNKNOWN)
        return None

    def fork(self, player):
        test = 1 if player == self.USER else 4
        threats = []
        for i in range(3):
            if self.row_sum[i] == test:
                threats.extend((i, j) for j in range(3) if self.board[i][j] == self.UNKNOWN)
            if self.col_sum[i] == test:
                threats.extend((j, i) for j in range(3) if self.board[j][i] == self.UNKNOWN)
        if self.diag_a_sum == test:
            threats.extend(pos for pos in self.DIAGONAL_A if self.board[pos[0]][pos[1]] == self.UNKNOWN)
        if self.diag_b_sum == test:
            threats.extend(pos for pos in self.DIAGONAL_B if self.board[pos[0]][pos[1]] == self.UNKNOWN)
        return next((threat for threat in threats if threats.count(threat) > 1), None)

    def two_in_a_row(self, block_fork):
        for i in range(3):
            if self.row_sum[i] == 4:
                move = next((j for j in range(3) if self.board[i][j] == self.UNKNOWN), None)
                if move and (i, move) == block_fork:
                    return i, move
            if self.col_sum[i] == 4:
                move = next((j for j in range(3) if self.board[j][i] == self.UNKNOWN), None)
                if move and (move, i) == block_fork:
                    return move, i
        for diag, diag_sum in [(self.DIAGONAL_A, self.diag_a_sum), (self.DIAGONAL_B, self.diag_b_sum)]:
            if diag_sum == 4:
                move = next((pos for pos in diag if self.board[pos[0]][pos[1]] == self.UNKNOWN), None)
                if move and move == block_fork:
                    return move
        return None

    def get_best_move(self):
        # Implement the perfect play strategy
        win = self.potential_win_check(self.COMPUTER)
        if win:
            return win
        block = self.potential_win_check(self.USER)
        if block:
            return block
        fork = self.fork(self.COMPUTER)
        if fork:
            return fork
        block_fork = self.fork(self.USER)
        if block_fork:
            return self.two_in_a_row(block_fork) or block_fork
        if self.board[1][1] == self.UNKNOWN:
            return (1, 1)
        for corner, opp in [((0,0), (2,2)), ((0,2), (2,0))]:
            if self.board[corner[0]][corner[1]] == self.USER and self.board[opp[0]][opp[1]] == self.UNKNOWN:
                return opp
        for corner in [(0,0), (0,2), (2,0), (2,2)]:
            if self.board[corner[0]][corner[1]] == self.UNKNOWN:
                return corner
        return next(((i, j) for i in range(3) for j in range(3) if self.board[i][j] == self.UNKNOWN), None)

    def play_game(self, perfect_player=None):
        moves = []
        current_player = self.COMPUTER if random.choice([True, False]) else self.USER
        while not self.check_winner() and not self.is_full():
            if (perfect_player is None) or (current_player != perfect_player):
                move = random.choice(self.get_empty_cells())
            else:
                move = self.get_best_move()
            self.put_mark(current_player, move)
            moves.append(f"{['X', 'O'][current_player-1]}{move[0]+1}{move[1]+1}")
            current_player = self.USER if current_player == self.COMPUTER else self.COMPUTER
        return moves

def generate_mixed_games(num_games=5000):
    perfect_games = [TicTacToe().play_game(perfect_player=TicTacToe.COMPUTER) for _ in range(num_games // 2)]
    random_games = [TicTacToe().play_game() for _ in range(num_games - len(perfect_games))]
    
    all_games = perfect_games + random_games
    random.shuffle(all_games)
    
    print(f"Nombre total de parties générées : {len(all_games)}")
    print(f"Nombre de parties parfaites : {len(perfect_games)}")
    print(f"Nombre de parties aléatoires : {len(random_games)}")
    
    for i, game in enumerate(all_games[:10]):
        print(f"Partie {i+1}: {' '.join(game)}")
    
    return all_games

# Chargement du modèle
out_dir = 'out-txt-models/'
device = 'cpu'
ckpt_path = os.path.join(out_dir, 'ckpt_iter_3200.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Extraction des activations
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach())

last_mlp = model.transformer.h[-1].mlp
last_mlp.register_forward_hook(hook_fn)

# Génération des parties et des activations
mixed_games = generate_mixed_games(10000)
all_activations = []

for game in mixed_games:
    encoded_game = encode_game(game)
    for i in range(len(encoded_game)):
        input_ids = torch.tensor(encoded_game[:i+1], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            model(input_ids)
        activation = activations[-1].squeeze()
        all_activations.append(activation if activation.dim() == 2 else activation.unsqueeze(0))
    activations.clear()

# Padding des activations
max_len = max(tensor.size(0) for tensor in all_activations)
padded_activations = [torch.cat([tensor, torch.zeros(max_len - tensor.size(0), tensor.size(1), device=device)], dim=0) 
                      if tensor.size(0) < max_len else tensor for tensor in all_activations]

# Conversion en tensor final
all_activations = torch.stack(padded_activations)
print(f"Shape of all_activations: {all_activations.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Détection du device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, sparsity_target=0.05, sparsity_weight=0.1, l1_weight=1e-5, l2_weight=1e-5):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        
        # Initialisation personnalisée
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_function(self, x, decoded, encoded):
        # Perte de reconstruction
        reconstruction_loss = F.mse_loss(decoded, x)
        
        # Perte de sparsité (KL divergence)
        p_hat = torch.mean(encoded, dim=0)
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, p_hat))
        
        # Régularisation L1
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.parameters())
        
        # Régularisation L2
        l2_loss = sum(torch.sum(param.pow(2)) for param in self.parameters())
        
        # Perte totale
        total_loss = reconstruction_loss + self.sparsity_weight * sparsity_loss + self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'sparsity': sparsity_loss.item(),
            'l1': l1_loss.item(),
            'l2': l2_loss.item()
        }

    @staticmethod
    def kl_divergence(p, p_hat):
        epsilon = 1e-10
        p_hat = torch.clamp(p_hat, epsilon, 1 - epsilon)
        return p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Exemple d'utilisation :
input_dim = all_activations.shape[2]  # À ajuster selon la dimension de vos données
encoding_dim = 1024
sparsity_target = 0.05
sparsity_weight = 0.0001  # Réduit de 0.1 à 0.0001
l1_weight = 1e-5  # Réduit de 1e-5 à 1e-6
l2_weight = 1e-6
num_epochs = 1000
batch_size = 1024

# Normalisation des données
mean = all_activations.mean()
std = all_activations.std()
all_activations = (all_activations - mean) / (std + 1e-10)

# Assurez-vous que all_activations est sur le bon device
all_activations = all_activations.to(device)

sae = SparseAutoencoder(input_dim, encoding_dim, sparsity_target, sparsity_weight, l1_weight, l2_weight).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-5)  # Taux d'apprentissage réduit à 1e-5
scaler = GradScaler()

# Boucle d'entraînement
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, all_activations.shape[0], batch_size):
        batch = all_activations[i:i+batch_size]
        optimizer.zero_grad()
        
        with autocast():
            encoded, decoded = sae(batch)
            if check_nan(encoded, "encoded") or check_nan(decoded, "decoded"):
                continue
            loss, loss_components = sae.loss_function(batch, decoded, encoded)
            if check_nan(loss, "loss"):
                print(f"Reconstruction: {loss_components['reconstruction']:.4f}, "
                      f"Sparsity: {loss_components['sparsity']:.4f}, "
                      f"L1: {loss_components['l1']:.4f}, "
                      f"L2: {loss_components['l2']:.4f}")
                continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Average Loss: {total_loss/all_activations.shape[0]:.4f}, '
              f'Reconstruction: {loss_components["reconstruction"]:.4f}, '
              f'Sparsity: {loss_components["sparsity"]:.4f}, '
              f'L1: {loss_components["l1"]:.4f}, '
              f'L2: {loss_components["l2"]:.4f}')

# Sauvegarder le modèle
torch.save(sae.state_dict(), 'sae_model.pth')