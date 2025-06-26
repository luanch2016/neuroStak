from fastapi import FastAPI
from schemas.network import NetworkDefinition
from models.builder import build_model, generate_code

app = FastAPI()

@app.post("/build-model/")
def build_pytorch_model(net_def: NetworkDefinition):
    model = build_model(net_def)
    code = generate_code(net_def)
    return {"message": "Model built successfully", "code": code}