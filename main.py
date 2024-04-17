from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

templates = Jinja2Templates(directory="templates")  # html 코드를 사용하기 위한 코드

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # css, js 코드를 사용하기 위한 코드


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

