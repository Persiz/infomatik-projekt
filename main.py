from loguru import logger
logger.info("Importing libraries")
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# Trainingsdaten runterladen
logger.info("Loading datasets")    
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Testdaten runterladen
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
logger.info("done")



# DataLoader erstellen
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break





device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Erstelle verschiedene neurale Netze
class OneLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return self.classifier(x)

# Modelle definieren
models = {
    'OneLayerMLP': OneLayerMLP().to(device),
    'DeepMLP': DeepMLP().to(device),
    'SimpleCNN': SimpleCNN().to(device)
}

for name, model in models.items():
    print(f"\n{name} Model:")
    print(model)

loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Berechne Vorhersage und Verlust
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / len(dataloader)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

# Trainiere alle Modelle
epochs = 5
results = {}

for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Trainiere {model_name} Model")
    print(f"{'='*50}\n")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, accuracy = test(test_dataloader, model, loss_fn)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy * 100)
    
    results[model_name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }
    
    # Speichere das Modell
    torch.save(model.state_dict(), f"model_{model_name.lower()}.pth")
    print(f"Modell gespeichert als 'model_{model_name.lower()}.pth'")

print("Done!")
logger.info("Training complete")

# Visualisierung / Vergleich aller Modelle
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Test Loss Vergleich
for model_name, result in results.items():
    axes[0, 0].plot(range(1, epochs + 1), result['test_losses'], marker='o', label=model_name, linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Test Loss', fontsize=12)
axes[0, 0].set_title('Test Loss Vergleich', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Test Accuracy Vergleich
for model_name, result in results.items():
    axes[0, 1].plot(range(1, epochs + 1), result['test_accuracies'], marker='o', label=model_name, linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 1].set_title('Test Accuracy Vergleich', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 100])

# Plot 3: Training Loss Vergleich
for model_name, result in results.items():
    axes[1, 0].plot(range(1, epochs + 1), result['train_losses'], marker='s', label=model_name, linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Training Loss', fontsize=12)
axes[1, 0].set_title('Training Loss Vergleich', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final Accuracy Bar Chart
final_accuracies = [results[model_name]['test_accuracies'][-1] for model_name in results.keys()]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 1].bar(results.keys(), final_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Final Accuracy (%)', fontsize=12)
axes[1, 1].set_title('Finale Accuracy nach Training', fontsize=14, fontweight='bold')
axes[1, 1].set_ylim([0, 100])
for i, (name, acc) in enumerate(zip(results.keys(), final_accuracies)):
    axes[1, 1].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("Vergleichsgraph gespeichert als 'model_comparison.png'")

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')