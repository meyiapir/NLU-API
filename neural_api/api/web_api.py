from fastapi import FastAPI
from neural_api.nlu.ml_module import nlu_handler
import os

app = FastAPI()

@app.get("/get_answer")
async def get_answer(message: str):
    return {"data": nlu_handler(message)}

@app.get("/check_service")
async def check_service():
    return {"data": "ok"}


if __name__ == "__main__":
    command = 'uvicorn web_api:app --host 127.0.0.1 --port 8000'
    os.system(command)
