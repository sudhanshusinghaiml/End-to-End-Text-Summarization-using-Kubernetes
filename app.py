from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.text_summarization.pipeline.prediction_pipeline import PredictionPipeline


text:str = "What is Text Summarization?"

TextSummarizationApp = FastAPI()

@TextSummarizationApp.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@TextSummarizationApp.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    


@TextSummarizationApp.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    



if __name__=="__main__":
    uvicorn.run(TextSummarizationApp, host="0.0.0.0", port=8080)
