import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

class PNGDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, code_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_function(self, x, decoded, encoded, sparsity_param=1e-3, sparsity_weight=1e-3):
        mse_loss = nn.MSELoss()(decoded, x)
        avg_activation = torch.mean(encoded, dim=0)
        kl_div = torch.sum(sparsity_param * torch.log(sparsity_param / (avg_activation + 1e-8)) +
                           (1 - sparsity_param) * torch.log((1 - sparsity_param) / (1 - avg_activation + 1e-8)))
        total_loss = mse_loss + sparsity_weight * kl_div
        return total_loss

# Paramètres
input_size = 81  # 9x9 images
hidden_size = 64
code_size = 32
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
dataset_path = 'data/img/files'  # Chemin vers le dossier contenant les images PNG

# Préparation du dataset
transform = transforms.Compose([
    transforms.Resize((9, 9)),
    transforms.ToTensor(),
])

dataset = PNGDataset(dataset_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialisation du modèle et de l'optimiseur
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = SparseAutoencoder(input_size, hidden_size, code_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        data = batch.view(batch.size(0), -1).to(device)
        optimizer.zero_grad()
        encoded, decoded = model(data)
        loss = model.loss_function(data, decoded, encoded)
        if not torch.isfinite(loss):
            print(f"Warning: Loss is {loss.item()}, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Perte moyenne: {avg_loss:.4f}')

# Visualisation des résultats
model.eval()
with torch.no_grad():
    test_img = next(iter(train_loader)).to(device)
    test_img_flat = test_img.view(1, -1)
    _, reconstructed = model(test_img_flat)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(test_img.cpu().squeeze(), cmap='gray')
    plt.title('Image originale')
    plt.subplot(122)
    plt.imshow(reconstructed.cpu().view(9, 9), cmap='gray')
    plt.title('Image reconstruite')
    plt.show()

print("Entraînement terminé et résultats visualisés.")