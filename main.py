import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

DATA_FOLDER = "data"
BATCH_SIZE = 250
LEARNING_RATE = 0.01
EPOCHS = 100


class Model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn,
    device: str = "cpu",
) -> tuple[float, float]:
    train_loss, train_accuracy = 0, 0

    model.train()
    for X_train, y_train in data_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)

        y_preds = model(X_train)

        loss = loss_fn(y_preds, y_train)

        train_loss += loss.item()
        train_accuracy += accuracy_fn(y_preds.argmax(dim=1), y_train).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)

    return train_loss, train_accuracy


def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: str = "cpu",
) -> tuple[float, float]:
    test_loss, test_accuracy = 0, 0

    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_y_preds = model(X_test)

            test_loss += loss_fn(test_y_preds, y_test).item()
            test_accuracy += accuracy_fn(test_y_preds.argmax(dim=1), y_test).item()

        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)

    return test_loss, test_accuracy


def get_data_loaders() -> tuple[DataLoader, DataLoader]:
    train_data = MNIST(
        root=DATA_FOLDER,
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None,
    )

    test_data = MNIST(
        root=DATA_FOLDER,
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=None,
    )

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_data_loader, test_data_loader


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data_loader, test_data_loader = get_data_loaders()

    input_shape = 28 * 28
    hidden_units = 10
    output_shape = 10

    model = Model(input_shape, hidden_units, output_shape).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    accuracy_fn = MulticlassAccuracy(num_classes=10).to(device)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_step(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            accuracy_fn,
            device,
        )

        test_loss, test_accuracy = test_step(
            model,
            test_data_loader,
            loss_fn,
            accuracy_fn,
            device,
        )

        print(
            f"Epoch: {epoch} | Loss: {train_loss:.4f} Accuracy: {train_accuracy * 100:.2f}% | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
