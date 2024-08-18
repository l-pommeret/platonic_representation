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

    # Dictionnaire pour les couleurs des joueurs
    color_map = {'X': X, 'O': O}

    # Parcourir les mouvements pour dessiner les pièces
    for i, move in enumerate(moves):
        player = 'X' if i % 2 == 0 else 'O'
        row, col = int(move[1]) - 1, int(move[2]) - 1

        # Calculer la position sur l'image
        pixel_x = col + (i % 3) * 3
        pixel_y = row + (i // 3) * 3

        # Dessiner la pièce
        draw.point((pixel_x, pixel_y), fill=color_map[player])

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