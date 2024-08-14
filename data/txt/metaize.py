import pickle

vocab = {
    'vocab_size': 7,
    'itos': {0: ' ', 1: '1', 2: '2', 3: '3', 4: 'O', 5: 'X', 6: '\n'},
    'stoi': {' ': 0, '1': 1, '2': 2, '3': 3, 'O': 4, 'X': 5, '\n': 6}
}

# Écriture du dictionnaire dans un fichier pickle
with open('data/txt/meta.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Le fichier meta.pkl a été créé avec succès.")

# Pour vérifier le contenu (optionnel)
with open('data/txt/meta.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
print("Contenu chargé :", loaded_vocab)