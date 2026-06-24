from loguru import logger
logger.info("Importing libraries")
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("Streamlit ist nicht runtergeladen.")
    sys.exit(1)

from PIL import Image
import numpy as np

classes = [str(i) for i in range(10)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model_classes = {
    'OneLayerMLP': OneLayerMLP,
    'DeepMLP': DeepMLP,
    'SimpleCNN': SimpleCNN,
}

@st.cache_data
def load_datasets():
    training_data = datasets.MNIST(
        root='data', train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root='data', train=False, download=True, transform=ToTensor()
    )
    return training_data, test_data

@st.cache_resource
def create_model(name):
    model = model_classes[name]().to(device)
    return model

def model_path(name: str) -> str:
    return f"model_{name.lower()}.pth"

def weights_available() -> bool:
    return all(os.path.exists(model_path(name)) for name in model_classes)

@st.cache_resource
def load_models():
    models = {}
    for name in model_classes:
        model = create_model(name)
        model.load_state_dict(torch.load(model_path(name), map_location=device))
        model.eval()
        models[name] = model
    return models


def train(dataloader, model, loss_fn, optimizer, progress_callback=None):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if progress_callback is not None:
            progress_callback((batch + 1) / len(dataloader))
    return total_loss / len(dataloader)


def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    total_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return total_loss, accuracy * 100


def train_all_models(epochs: int):
    training_data, test_data = load_datasets()
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    results = {}
    trained_models = {}
    loss_fn = nn.CrossEntropyLoss()

    for name in model_classes:
        model = model_classes[name]().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        train_losses = []
        test_losses = []
        test_accuracies = []
        progress_bar = st.progress(0.0, text=f"Trainiere {name}")

        for epoch in range(epochs):
            epoch_loss = train(
                train_dataloader,
                model,
                loss_fn,
                optimizer,
                progress_callback=lambda progress, pb=progress_bar: pb.progress(progress),
            )
            test_loss, accuracy = evaluate(test_dataloader, model, loss_fn)
            train_losses.append(epoch_loss)
            test_losses.append(test_loss)
            test_accuracies.append(accuracy)

        torch.save(model.state_dict(), model_path(name))
        trained_models[name] = model
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
        }
        progress_bar.empty()

    create_model.clear()
    load_models.clear()
    return results, trained_models


def ensure_models_trained(epochs: int):
    if st.session_state.training_results is None:
        with st.spinner('Modelle werden trainiert...'):
            results, _ = train_all_models(epochs)
            st.session_state.training_results = results
            st.success('Training abgeschlossen.')
        return results
    return st.session_state.training_results


def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = len(next(iter(results.values()))['train_losses'])

    for name, result in results.items():
        axes[0, 0].plot(range(1, epochs + 1), result['test_losses'], marker='o', label=name)
    axes[0, 0].set_title('Test Loss Vergleich')
    axes[0, 0].set_xlabel('Epochen')
    axes[0, 0].set_ylabel('Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for name, result in results.items():
        axes[0, 1].plot(range(1, epochs + 1), result['test_accuracies'], marker='o', label=name)
    axes[0, 1].set_title('Test Accuracy Vergleich')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 100])

    for name, result in results.items():
        axes[1, 0].plot(range(1, epochs + 1), result['train_losses'], marker='s', label=name)
    axes[1, 0].set_title('Training Loss Vergleich')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    final_accuracies = [results[name]['test_accuracies'][-1] for name in results]
    axes[1, 1].bar(results.keys(), final_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Finale Test Accuracy')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_ylim([0, 100])
    for i, acc in enumerate(final_accuracies):
        axes[1, 1].text(i, acc + 1, f"{acc:.1f}%", ha='center')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    return fig


def preprocess_canvas(image_data):
    image = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    image = image.convert('L')
    image = image.resize((28, 28))
    array = np.array(image) / 255.0
    array = 1 - array
    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, array


def get_predictions(img_tensor):
    models = load_models()
    predictions = {}
    softmax = torch.nn.Softmax(dim=1)
    for name, model in models.items():
        with torch.no_grad():
            pred = model(img_tensor)
            probs = softmax(pred)[0].cpu().numpy()
            label = classes[int(probs.argmax())]
            predictions[name] = {
                'label': label,
                'confidence': float(probs.max()),
                'probabilities': probs.tolist(),
            }
    return predictions


def app():
    st.set_page_config(page_title='MNIST Browser GUI', layout='wide')
    st.title('MNIST Zeichen-App')
    st.write('Trainiere ein Modell, zeichne eine Zahl und lasse sie erraten.')

    training_data, test_data = load_datasets()
    st.sidebar.header('Einstellungen')
    epochs = st.sidebar.slider('Epochen', 1, 10, 5)
    st.sidebar.write(f'Device: {device}')

    if weights_available():
        st.sidebar.success('Gewichte vorhanden: model_onelayermlp.pth, model_deepmlp.pth, model_simplecnn.pth')
    else:
        st.sidebar.warning('Keine Gewichte vorhanden, trainiere vor dem Testen.')

    if 'training_results' not in st.session_state:
        st.session_state.training_results = None

    if st.sidebar.button('Trainiere alle Modelle'):
        with st.spinner('Modelle werden trainiert...'):
            results, models = train_all_models(epochs)
            st.session_state.training_results = results
            st.success('Training abgeschlossen!')
            st.rerun()

    if st.session_state.training_results is not None:
        results = st.session_state.training_results
    else:
        results = None

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader('Zeichne hier')
        canvas_result = st_canvas(
            fill_color='rgba(0, 0, 0, 0)',
            stroke_width=20,
            stroke_color='#000000',
            background_color='#FFFFFF',
            height=280,
            width=280,
            drawing_mode='freedraw',
            key='canvas',
        )

        if st.button('Erkennen'):
            if canvas_result.image_data is not None:
                img_tensor, img_array = preprocess_canvas(canvas_result.image_data)
                if st.session_state.training_results is None:
                    ensure_models_trained(epochs)
                    results = st.session_state.training_results
                predictions = get_predictions(img_tensor)
                st.image(img_array, caption='Verarbeitetes Bild', clamp=True, width=250)

                for name, info in predictions.items():
                    st.metric(f'{name}', f'{info["label"]}', delta=f'{info["confidence"]:.2f}')

                st.subheader('Wahrscheinlichkeiten pro Modell')
                st.dataframe({name: info['probabilities'] for name, info in predictions.items()})
            else:
                st.warning('Bitte zeichne zuerst eine Zahl!')

    with col2:
        st.subheader('Modelle und Evaluation')
        if results is not None:
            st.pyplot(plot_results(results))
            st.write('Finale Genauigkeiten:')
            final_acc = {name: result['test_accuracies'][-1] for name, result in results.items()}
            st.table(final_acc)
        elif weights_available():
            st.info('Modelle sind geladen. Sie werden vor der ersten Erkennung trainiert.')
        else:
            st.info('Keine Gewichte vorhanden, trainiere die Modelle in der Sidebar.')

    with st.expander('Modellarchitektur'):
        for name, cls in model_classes.items():
            st.markdown(f'**{name}**')
            st.text(str(cls()))


def is_running_in_streamlit() -> bool:
    try:
        return st.runtime.exists()
    except Exception:
        return False

if __name__ == '__main__':
    if not is_running_in_streamlit():
        if STREAMLIT_AVAILABLE:
            os.execvp(sys.executable, [sys.executable, '-m', 'streamlit', 'run', __file__])
        else:
            print('Bitte installiere streamlit und streamlit-drawable-canvas:')
            print('uv run python -m pip install streamlit streamlit-drawable-canvas')
            sys.exit(1)
    else:
        app()
