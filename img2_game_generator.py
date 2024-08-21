import csv
from PIL import Image, ImageDraw

from PIL import Image, ImageDraw

def create_board_image(moves):
    # Définir les couleurs
    EMPTY = 128  # Gris
    O = 0        # Noir
    X = 255      # Blanc

    # Créer une image 9x9
    img = Image.new('L', (9, 9), color=EMPTY)
    draw = ImageDraw.Draw(img)

    # Fonction pour dessiner un pion
    def draw_pion(x, y, player):
        color = X if player == 'X' else O
        draw.point((x, y), fill=color)

    # Traiter chaque coup
    for i, move in enumerate(moves):
        player = 'X' if i % 2 == 0 else 'O'
        row, col = int(move[1]) - 1, int(move[2]) - 1

        # Calculer la position du pion dans l'image 9x9
        x = (i % 3) * 3 + col
        y = (i // 3) * 3 + row

        # Dessiner le pion
        draw_pion(x, y, player)

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
output_prefix = 'data/img2/files/game'
generate_game_images(csv_file, output_prefix)
print("Génération d'images terminée.")


