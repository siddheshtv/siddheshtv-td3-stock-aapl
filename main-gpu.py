import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import logging
import matplotlib.pyplot as plt
from collections import deque


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        return self.max_action * torch.tanh(self.layer_3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.layer_1(sa))
        q = torch.relu(self.layer_2(q))
        return self.layer_3(q)

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()))

        self.max_action = max_action
        self.total_it = 0
        self.actor_loss = deque(maxlen=100)
        self.critic_loss = deque(maxlen=100)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            
            batch_states, batch_actions, batch_rewards, batch_next_states = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(batch_states).to(device)
            action = torch.FloatTensor(batch_actions).to(device)
            reward = torch.FloatTensor(batch_rewards).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(batch_next_states).to(device)

            
            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            
            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_Q).detach()

            
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = nn.functional.mse_loss(current_Q1, target_Q)
            
            
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = nn.functional.mse_loss(current_Q2, target_Q)

            
            critic_loss = loss_Q1 + loss_Q2
            self.critic_loss.append(critic_loss.item())

            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            
            if it % policy_freq == 0:
                
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                self.actor_loss.append(actor_loss.item())
                
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.total_it += 1

            if self.total_it % 1000 == 0:
                logging.info(f"Iteration: {self.total_it}, Actor Loss: {np.mean(self.actor_loss):.4f}, Critic Loss: {np.mean(self.critic_loss):.4f}")

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, action, reward, next_state)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, action, reward, next_state))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states = [], [], [], []
        for i in ind: 
            state, action, reward, next_state = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
        return np.array(batch_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_next_states)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def train_td3_stock_prediction(stock_data, epochs=1000, batch_size=100):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    state_dim = scaled_data.shape[1]
    action_dim = 1
    max_action = 1

    td3 = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    episode_reward = []
    episode_sharpe_ratio = []

    for epoch in range(epochs):
        state = scaled_data[0]
        episode_return = 0
        returns = []

        for t in range(1, len(scaled_data)):
            action = td3.select_action(state)
            next_state = scaled_data[t]
            reward = -abs(action - next_state[0])  
            replay_buffer.add(state, action, reward, next_state)
            
            episode_return += reward
            returns.append(reward)
            
            state = next_state

            if len(replay_buffer.storage) > batch_size:
                td3.train(replay_buffer, 10, batch_size)

        episode_reward.append(episode_return)
        sharpe = calculate_sharpe_ratio(np.array(returns))
        episode_sharpe_ratio.append(sharpe)

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs}, Reward: {episode_return:.4f}, Sharpe Ratio: {sharpe:.4f}")

        if (epoch + 1) % 100 == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(episode_reward)
            plt.title('Episode Rewards')
            plt.subplot(2, 1, 2)
            plt.plot(episode_sharpe_ratio)
            plt.title('Episode Sharpe Ratios')
            plt.tight_layout()
            plt.savefig(f'training_metrics_epoch_{epoch+1}.png')
            plt.close()

    return td3, episode_reward, episode_sharpe_ratio

def load_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].values


if __name__ == "__main__":
    ticker = "AAPL"  
    start_date = "2020-01-01"
    end_date = "2023-12-31"

    stock_data = load_stock_data(ticker, start_date, end_date)
    trained_model, rewards, sharpe_ratios = train_td3_stock_prediction(stock_data)

    
    torch.save(trained_model.actor.state_dict(), f'td3_stock_prediction_model_{ticker}.pth')

    logging.info(f"Model training completed and saved for {ticker}.")

    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('Final Episode Rewards')
    plt.subplot(2, 1, 2)
    plt.plot(sharpe_ratios)
    plt.title('Final Episode Sharpe Ratios')
    plt.tight_layout()
    plt.savefig('final_training_metrics.png')
    plt.close()

    logging.info("Final metrics plot saved as 'final_training_metrics.png'")

###1 To load and use the model later:
###1 state_dim = ...  
###1 action_dim = 1
###1 max_action = 1
###1 loaded_model = Actor(state_dim, action_dim, max_action).to(device)
###1 loaded_model.load_state_dict(torch.load(f'td3_stock_prediction_model_{ticker}.pth'))
###1 loaded_model.eval()

###1 Example of using the loaded model for prediction:
###1 def predict_next_price(model, current_state):
###1     with torch.no_grad():
###1        current_state = torch.FloatTensor(current_state).unsqueeze(0).to(device)
###1         predicted_action = model(current_state)
###1     return predicted_action.cpu().item()

###1 current_state = ...  
###1 next_price_prediction = predict_next_price(loaded_model, current_state)
###1 print(f"Predicted next price movement: {next_price_prediction}")