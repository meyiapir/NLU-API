import os
from fastapi import FastAPI
if os.getenv("INDOCKER") is None:
    from neural_api.nlu.ml_module import nlu_handler
# else:
#     from ml_module import nlu_handler

app = FastAPI()



@app.get("/get_answer")
async def get_answer(message: str):
    return {"data": nlu_handler(message)}

@app.get("/check_service")
async def check_service():
    return {"data": "ok"}

#
# if os.getenv("INDOCKER") is None:
#     os.system('uvicorn web_api:app --host 0.0.0.0 --port 8001')
print("API is running...")
