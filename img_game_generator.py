import csv
from PIL import Image, ImageDraw

def create_board_image(moves):
    # Définir les couleurs
    EMPTY = 128  # Gris
    O = 0  # Noir
    X = 255  # Blanc

    # Créer une image 9x9
    img = Image.new('L', (9, 9), color=EMPTY)
    draw = ImageDraw.Draw(img)

    # Fonction pour dessiner un carré 3x3
    def draw_3x3(start_x, start_y, board):
        for i in range(3):
            for j in range(3):
                color = EMPTY
                if board[i][j] == 'O':
                    color = O
                elif board[i][j] == 'X':
                    color = X
                draw.point((start_x + j, start_y + i), fill=color)

    # Initialiser le plateau
    board = [[' ' for _ in range(3)] for _ in range(3)]

    # Remplir le plateau selon les mouvements
    for i, move in enumerate(moves):
        player = 'X' if i % 2 == 0 else 'O'
        row, col = int(move[1]) - 1, int(move[2]) - 1
        board[row][col] = player

        # Calculer la position du sous-plateau 3x3
        sub_x = (i % 3) * 3
        sub_y = (i // 3) * 3

        # Dessiner le sous-plateau 3x3
        draw_3x3(sub_x, sub_y, board)

    return img

def generate_game_images(csv_file, output_prefix):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Ignorer l'en-tête
        for i, row in enumerate(reader):
            moves = row[0].split()
            img = create_board_image(moves)
            img.save(f"{output_prefix}_{i+1}.png")
            if i % 10000 == 0:
                print(f"Généré {i+1} images")

# Utilisation
csv_file = 'data/txt/all_tic_tac_toe_games.csv'
output_prefix = 'data/img/files/game'
generate_game_images(csv_file, output_prefix)
print("Génération d'images terminée.")


