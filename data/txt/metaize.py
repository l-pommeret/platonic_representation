import pickle

vocab = {
    'vocab_size': 12,
    'itos': {0: ' ', 1: '0', 2: '1', 3: '2', 4: '3', 5: 'X', 6: 'O', 7: '/', 8: '-'},
    'stoi': {' ': 0, '0': 1, '1': 2, '2': 3, '3': 4, 'X': 5, 'O': 6, '/': 7, '-': 8}
}

# Écriture du dictionnaire dans un fichier pickle
with open('data/txt/meta.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Le fichier meta.pkl a été créé avec succès.")

# Pour vérifier le contenu (optionnel)
with open('data/txt/meta.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
print("Contenu chargé :", loaded_vocab)