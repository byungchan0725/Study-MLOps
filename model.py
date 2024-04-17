import argparse
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from setup import mkdir_folder, download_mnist_dataset
from train import train
from test import test

# 하이퍼파라미터
BATCH_SIZE = 100
EPOCH = 20
LR = 1e-3

transform = transforms.Compose([transforms.ToTensor()])


class ConvNet(nn.Module):  # 모델 정의
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU())
        self.Maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        self.Maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.Maxpool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 3 * 3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Maxpool1(x)
        x = self.conv2(x)
        x = self.Maxpool2(x)
        x = self.conv3(x)
        x = self.Maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    global train_dl, test_dl

    parser = argparse.ArgumentParser(description="model train과 test 설정")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--test", action="store_true", help="Test mode")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = ConvNet()
    model = model.to(device)

    mkdir_folder()
    train_dl, test_dl = download_mnist_dataset(transform, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    if args.train:
        print(f"""
        학습을 시작합니다.
        device: {device}
        Epoch: {EPOCH}, LR: {LR}
        Dataset size: {len(train_dl)}, Batch size: {BATCH_SIZE}
        Dataset name: MNIST DATASET
        """)

        loss_history = train(model, train_dl, criterion, optimizer, EPOCH, device)

    if args.test:
        try:
            road_model = torch.load("./train_result/CNN_MNIST.pt")

        except FileNotFoundError:
            print('학습된 모델이 없어 학습을 시작합니다.')
            train(model, train_dl, criterion, optimizer, EPOCH, device)
            road_model = torch.load("./train_result/CNN_MNIST.pt")

        finally:
            print(f"""
                    테스트를 시작합니다.
                    device: {device}
                    Dataset name: MNIST DATASET
                    """)

            test(road_model, test_dl, device)

