import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory="templates")  # HTML 코드를 사용하기 위한 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # CSS, JS 코드를 사용하기 위한 설정


# 모델 정의
class ConvNet(nn.Module):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 홈 페이지 라우팅
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 이미지 예측 핸들러
@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    model = ConvNet()  # 모델 인스턴스 생성
    model.to(device)
    model.eval()

    # 이미지를 PIL 이미지로 열기
    img = Image.open(io.BytesIO(await image.read()))

    # 이미지를 모델이 처리할 수 있는 형식으로 변환 (예: RGB, 텐서로 변환 등)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 이미지 크기 조정
        transforms.Grayscale(),
        transforms.ToTensor(),         # 텐서로 변환
    ])
    img = transform(img).unsqueeze(0)  # 배치 차원 추가

    # 모델에 이미지 전달하여 예측 수행
    output = model(img)

    # 예측 결과 출력
    prediction = output.argmax(dim=1).item()
    return {"prediction": prediction}
