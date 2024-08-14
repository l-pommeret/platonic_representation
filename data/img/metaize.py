import pickle

vocab = {
    'vocab_size': 4,
    'itos': {0: 'blanc', 1: 'noir', 2: 'gris', 3: 'début'},
    'stoi': {'blanc': 0, 'noir': 1, 'gris': 2, 'début': 3}
}

# Écriture du dictionnaire dans un fichier pickle
with open('data/img/meta.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Le fichier meta.pkl a été créé avec succès.")

# Pour vérifier le contenu (optionnel)
with open('data/img/meta.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
print("Contenu chargé :", loaded_vocab)