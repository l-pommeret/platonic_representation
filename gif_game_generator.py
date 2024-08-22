import csv
from PIL import Image, ImageDraw

def create_board_image(move, player, size=3):
    # Définir les couleurs
    EMPTY = 128  # Gris
    O = 0  # Noir
    X = 255  # Blanc
    # Créer une image 3x3
    img = Image.new('L', (size, size), color=EMPTY)
    draw = ImageDraw.Draw(img)
    # Dessiner le pion
    row, col = int(move[1]) - 1, int(move[2]) - 1
    color = X if player == 'X' else O
    draw.point((col, row), fill=color)
    return img

def generate_game_gifs(csv_file, output_prefix, max_games=100000):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Ignorer l'en-tête
        for i, row in enumerate(reader):
            if i >= max_games:
                break
            moves = row[0].split()
            images = []
            # Générer une image pour chaque coup
            for j, move in enumerate(moves):
                player = 'X' if j % 2 == 0 else 'O'
                img = create_board_image(move, player)
                images.append(img)
            # Créer et sauvegarder le GIF
            images[0].save(f"{output_prefix}_{i+1}.gif",
                           save_all=True,
                           append_images=images[1:],
                           duration=500,
                           loop=0)
            if i % 1000 == 0:
                print(f"Généré {i+1} GIFs")
    print("Génération de GIFs terminée.")

# Utilisation
csv_file = 'data/txt/all_tic_tac_toe_games.csv'
output_prefix = 'data/gif/files/game'
generate_game_gifs(csv_file, output_prefix)