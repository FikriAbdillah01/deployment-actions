import json
from fastapi import FastAPI, Request, Body
import numpy as np

app = FastAPI


@app.get("/")
async def read_main():
    return {"message":"Hello World"}

@app.get('/health')
async def health_check():
    return {"status":"healthy"}

