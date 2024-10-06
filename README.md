---
license: mit
language:
  - en
tags:
  - td3
  - stock prediction
  - reinforcement learning
---

# TD3 Model for AAPL Stock Prediction

## Model Description

This model is a TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm applied for stock price prediction, specifically trained on AAPL (Apple Inc.) stock data. The TD3 model is a reinforcement learning agent that interacts with a stock market environment and is trained to predict and maximize the return from trading AAPL stock.

### Key Features

- Asset: AAPL (Apple Inc.) Stock
- Model Type: TD3 (Twin Delayed DDPG)
- Action Space: Continuous (Buy, Sell, Hold decisions)
- Reward: Modeled on cumulative returns
- Training Data: Historical stock prices and related financial indicators for AAPL stock
- Environment: Custom stock trading environment simulating price movement and portfolio management
- Framework: PyTorch

## Quick Run

To use this model for stock prediction and trading, install the required dependencies and load the model via Hugging Face. Here is an example code snippet:

```python
import torch
from huggingface_hub import hf_hub_download
import torch.nn as nn
import numpy as np

# Download the model
model_path = hf_hub_download(repo_id="siddheshtv/td3-stock-aapl", filename="td3_stock_prediction_model_AAPL_full.pth")

# Load the model
checkpoint = torch.load(model_path)

# Recreate the Actor class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

# Instantiate the model
model = Actor(checkpoint['state_dim'], checkpoint['action_dim'], checkpoint['max_action'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Function to select action
def select_action(state):
    with torch.no_grad():
        state = torch.FloatTensor(state.reshape(1, -1))
        return model(state).cpu().data.numpy().flatten()

# Example usage
state = np.random.rand(checkpoint['state_dim'])  # Replace with actual state data
action = select_action(state)
print(f"Predicted action: {action}")
```

## Citation

```
@misc{siddheshtv-td3,
  title={TD3 Model for AAPL Stock Prediction},
  author={Siddhesh Kulthe},
  year={2024},
  howpublished={\url{https://huggingface.co/siddheshtv/td3-stock-aapl}},
  note={TD3 model for predicting stock price movements of AAPL using reinforcement learning},
}
```
