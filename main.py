import torch
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

templates = Jinja2Templates(directory="templates")  # html 코드를 사용하기 위한 코드

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # css, js 코드를 사용하기 위한 코드

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


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload")
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/result")
async def result(request: Request):
    image = Image.open("image.jpg")
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    return templates.TemplateResponse("result.html", {"request": request})
