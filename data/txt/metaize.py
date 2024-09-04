import pickle

vocab = {
    'vocab_size': 10,
    'itos': {0: ';', 1: ' ', 2: '0', 3: '1', 4: '2', 5: '3', 6: 'X', 7: 'O', 8: 'o', 9: 'x', 10: 'n'},
    'stoi': {';': 0, ' ': 1, '0': 2, '1': 3, '2': 4, '3': 5, 'X': 6, 'O': 7, 'o': 8, 'x': 9, 'n': 10}
}

# Écriture du dictionnaire dans un fichier pickle
with open('data/txt/meta.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Le fichier meta.pkl a été créé avec succès.")

# Pour vérifier le contenu (optionnel)
with open('data/txt/meta.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
print("Contenu chargé :", loaded_vocab)