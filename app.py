from fastapi import FastAPI
import uvicorn
import sys, os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.text_summarization.pipeline.prediction_pipeline import PredictionPipeline
from src.text_summarization.logger import logging


text:str = "What is Text Summarization?"

TextSummarizationApp = FastAPI()

@TextSummarizationApp.get("/", tags=["authentication"])
async def index():
    logging.info(f"Inside index() method routing get('/', tags=['authentication'])")
    return RedirectResponse(url="/docs")



@TextSummarizationApp.get("/train")
async def training():
    try:
        logging.info(f"Inside training() method routing get('/train')")
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as error:
        logging.exception(f"Error Occurred! {error}")
        return Response(f"Error Occurred! {error}")
    


@TextSummarizationApp.post("/predict")
async def predict_route(text):
    try:
        logging.info(f"Inside predict_route() method routing post('/predict')")
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as error:
        logging.exception(f"Error Occurred! {error}")
        raise Response(f"Error Occurred! {error}")
    


if __name__=="__main__":
    uvicorn.run(TextSummarizationApp, host="0.0.0.0", port=8080)

