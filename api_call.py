# import the necessary packages
from dataclasses import dataclass
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from app import main

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/gesture-recognizer/process-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        print("Image received")
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # if the image is not given
        if img is None:
            return JSONResponse(content={"result": "Image is not given"})

        # Call the main function from your app
        result = main(img)
        print(result)
        # json should be returned as follows
        # { "gesture": result[0]], "confidence": result[1]}
        return JSONResponse(content={"gesture": result[0], "confidence": result[1]})
    except ValueError as ve:
        # Handle ValueError and return an error response
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# api call
# uvicorn api_call:app --reload
