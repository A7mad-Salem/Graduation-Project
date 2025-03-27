%%writefile app.py

from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel

# Step 1: Define the model architecture (same as the trained model)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.7)
        self.fc = nn.Linear(128, 64)
        self.out = nn.Linear(64, 4)  # Change 4 to match the number of classes

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x

# Step 2: Load the trained model
model = Classifier()
model.load_state_dict(torch.load("DSC-BiLSTM_model.pth"))
model.eval()

# Step 3: Initialize FastAPI
app = FastAPI()

# Step 4: Define the Input Data Format
class InputData(BaseModel):
    features: list  # Expecting a list of 7 numerical features

# Step 5: Create the Prediction Endpoint
@app.post("/predict")
def predict(data: InputData):
    x_input = torch.tensor([data.features], dtype=torch.float32)
    with torch.no_grad():
        output = model(x_input)
        prediction = torch.argmax(output, dim=1).item()
    return {"prediction": prediction}

