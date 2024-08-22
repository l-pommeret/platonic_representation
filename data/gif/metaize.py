import pickle

vocab = {
    'vocab_size': 4,
    'itos': {0: 'b', 1: 'n', 2: 'g', 3: ';'},
    'stoi': {'b': 0, 'n': 1, 'g': 2, ';': 3}
}

# Écriture du dictionnaire dans un fichier pickle
with open('data/img/meta.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Le fichier meta.pkl a été créé avec succès.")

# Pour vérifier le contenu (optionnel)
with open('data/img/meta.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
print("Contenu chargé :", loaded_vocab)