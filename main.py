import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

DATA_FOLDER = "data"
BATCH_SIZE = 250
LEARNING_RATE = 0.01
EPOCHS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)


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


def main() -> None:
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

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_shape = 28 * 28
    hidden_units = 10
    output_shape = 10

    model = Model(input_shape, hidden_units, output_shape)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    accuracy = Accuracy(task="multiclass", num_classes=10)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = 0, 0

        for X_train, y_train in train_dataloader:
            model.train()

            y_preds = model(X_train)

            loss = loss_fn(y_preds, y_train)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy += accuracy(y_preds, y_train)

        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader)

        test_loss, test_accuracy = 0, 0

        model.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                test_y_preds = model(X_test)

                test_loss += loss_fn(test_y_preds, y_test)
                test_accuracy += accuracy(test_y_preds, y_test)

            test_loss /= len(test_dataloader)
            test_accuracy /= len(test_dataloader)

        print(
            f"Epoch: {epoch} | Loss: {train_loss:.4f} Accuracy: {train_accuracy * 100:.2f}% | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
