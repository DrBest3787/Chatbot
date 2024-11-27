import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter
batch_size = 64
learning_rate = 0.001
epochs = 10
num_classes = 10  # CIFAR-10 hat 10 Klassen

# Datenvorbereitung
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet benötigt größere Eingaben
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Transfer Learning: Vortrainiertes ResNet verwenden
base_model = models.resnet18(pretrained=True)

# Modifizieren des Modells für CIFAR-10
class CustomResNet(nn.Module):
    def __init__(self, base_model, num_classes, max_epochs):
        super(CustomResNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-2])  # Bis vor die letzte FC-Schicht
        self.additional_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 7 * 7, num_classes)
        self.max_epochs = max_epochs

    def forward(self, x, epoch):
        x = self.base(x)
        x = self.additional_conv(x)
        phase_factor = epoch / self.max_epochs  # Dynamische Dropout-Rate
        x = nn.functional.dropout(x, p=0.5 + 0.1 * phase_factor, training=self.training)
        x = torch.flatten(x, 1)  # Global Average Pooling durch Flatten ersetzt
        x = self.fc(x)
        return x

model = CustomResNet(base_model, num_classes, epochs)
model = model.to(device)

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Fine-Tuning aktivieren
for param in model.base.parameters():
    param.requires_grad = True  # Entfriert alle Parameter im ResNet-Basismodell

# Unterschiedliche Lernraten für Basismodell und neue Schichten
base_params = model.base.parameters()
new_params = model.fc.parameters()
optimizer = optim.Adam([
    {'params': base_params, 'lr': learning_rate * 0.1},  # Niedrigere Lernrate für Basismodell
    {'params': new_params, 'lr': learning_rate}          # Standard-Lernrate für neue Schichten
])

# Scheduler bleibt gleich
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mixed Precision Training
                outputs = model(images, epoch)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Scheduler aktualisieren
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

def evaluate():
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images, epoch=epochs - 1)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

    # Confusion-Matrix und Metriken
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":
    train()
    evaluate()
