# 기본적으로 필요한 폴더 생성 및 데이터를 다운로드 하는 설정 코드


import os

from torchvision import datasets
from torch.utils.data import DataLoader


def mkdir_folder():  # 모델에 대한 폴더 생성
    folder_name = 'train_result'  # 생성할 폴더 이름
    dataset_folder = 'dataset'  # 데이터셋을 다운로드할 폴더 이름

    pwd = os.getcwd()  # 현재 위치

    print(f'현재 위치: {pwd}')

    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
        print(f'현재 위치에 {dataset_folder} 폴더를 생성하였습니다.')

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(f'현재 위치에 {folder_name} 폴더를 생성하였습니다.')


def download_mnist_dataset(transform, batch_size):  # 데이터셋 다운로드 폴더 생성
    download_root = './dataset'

    train_ds = datasets.MNIST(  # ds = DataSet의 약자
        root=download_root,
        train=True,
        download=True,
        transform=transform
    )
    test_ds = datasets.MNIST(
        root=download_root,
        train=False,
        download=True,
        transform=transform
    )

    train_dl = DataLoader(  # dl = DataLoader의 약자
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dl, test_dl
