import itertools
import random
import csv

def is_winner(board, player):
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or \
           all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or \
       all(board[i][2-i] == player for i in range(3)):
        return True
    return False

def generate_all_games():
    all_games = []
    positions = list(itertools.product(range(3), repeat=2))
    
    def backtrack(board, moves, turn):
        if is_winner(board, 'X'):
            all_games.append((moves, '0-1'))
            return
        if is_winner(board, 'O'):
            all_games.append((moves, '1-0'))
            return
        if len(moves) == 9:
            all_games.append((moves, '1/2-1/2'))
            return
        
        player = 'X' if turn % 2 == 0 else 'O'
        for i, j in positions:
            if board[i][j] == ' ':
                new_board = [row[:] for row in board]
                new_board[i][j] = player
                new_moves = moves + [f"{player}{i+1}{j+1}"]
                backtrack(new_board, new_moves, turn + 1)
    
    initial_board = [[' ' for _ in range(3)] for _ in range(3)]
    backtrack(initial_board, [], 0)
    return all_games

# Générer toutes les parties possibles
all_games = generate_all_games()

# Mélanger aléatoirement toutes les parties
random.shuffle(all_games)

# Créer un fichier CSV avec toutes les parties mélangées
csv_filename = 'data/all_tic_tac_toe_games.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Écrire l'en-tête
    csv_writer.writerow(['transcript', 'Result'])
    
    # Écrire toutes les parties
    for moves, result in all_games:
        csv_writer.writerow([' '.join(moves), result])

print(f"Toutes les parties ont été générées, mélangées et sauvegardées dans '{csv_filename}'")
print(f"Nombre total de parties : {len(all_games)}")

# Afficher les 5 premières lignes du fichier CSV comme exemple
print("\nExemple des 5 premières lignes du fichier CSV :")
with open(csv_filename, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for i, row in enumerate(csv_reader):
        if i == 0:
            print("En-tête :", row)
        elif i <= 5:
            print(f"Ligne {i} :", row)
        else:
            break