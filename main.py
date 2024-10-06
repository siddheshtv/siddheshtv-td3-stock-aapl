import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time
from scipy import stats
from tabulate import tabulate
import ta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

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

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=256, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            batch_states, batch_actions, batch_rewards, batch_next_states = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(batch_states).to(device)
            action = torch.FloatTensor(batch_actions).to(device)
            reward = torch.FloatTensor(batch_rewards).to(device)
            next_state = torch.FloatTensor(batch_next_states).to(device)

            with torch.no_grad():
                noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.critic_target_1(next_state, next_action), self.critic_target_2(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (discount * target_Q)

            current_Q1, current_Q2 = self.critic_1(state, action), self.critic_2(state, action)

            critic_loss = nn.functional.mse_loss(current_Q1, target_Q) + nn.functional.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

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

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, state, action, reward, next_state):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, action, reward, next_state)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, action, reward, next_state))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in ind]
        return map(np.array, zip(*batch))

def add_technical_indicators(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df.dropna(inplace=True)
    return df

def normalize_state(state, history):
    return (state - history.mean()) / history.std()

def calculate_reward(action, next_price, current_price, risk_free_rate=0.02):
    returns = (next_price - current_price) / current_price
    portfolio_return = action * returns
    excess_return = portfolio_return - risk_free_rate / 252
    return excess_return

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def plot_training_metrics(rewards, sharpe_ratios, training_times):
    epochs = range(1, len(rewards) + 1)

    plt.figure(figsize=(12, 6))


    plt.subplot(1, 3, 1)
    plt.plot(epochs, rewards, label="Rewards")
    plt.title("Episode Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")


    plt.subplot(1, 3, 2)
    plt.plot(epochs, sharpe_ratios, label="Sharpe Ratios", color="orange")
    plt.title("Sharpe Ratios")
    plt.xlabel("Epoch")
    plt.ylabel("Sharpe Ratio")


    plt.subplot(1, 3, 3)
    plt.plot(epochs, training_times, label="Training Time", color="green")
    plt.title("Training Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")

    plt.tight_layout()
    plt.show()

def train_td3_stock_prediction(stock_data, epochs=100, batch_size=256):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stock_data)

    state_dim = scaled_data.shape[1]
    action_dim = 1
    max_action = 1

    print(f"State dimension: {state_dim}") 

    td3 = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(max_size=1e5)

    episode_reward = []
    episode_sharpe_ratio = []
    training_times = []
    all_actions = []

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        state = scaled_data[0]
        episode_return = 0
        returns = []
        epoch_actions = []

        for t in range(1, len(scaled_data)):
            action = td3.select_action(state)
            next_state = scaled_data[t]
            reward = calculate_reward(action, next_state[3], state[3])
            replay_buffer.add(state, action, reward, next_state)

            episode_return += reward
            returns.append(reward)
            epoch_actions.append(action)

            state = next_state

            if len(replay_buffer.storage) > batch_size:
                td3.train(replay_buffer, min(10, len(replay_buffer.storage) - 1), batch_size)

        episode_reward.append(float(episode_return))
        sharpe = calculate_sharpe_ratio(np.array(returns))
        episode_sharpe_ratio.append(float(sharpe))

        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        all_actions.append(epoch_actions)

        if (epoch + 1) % 10 == 0:
            total_time = time.time() - start_time
            logging.info(f"Epoch {epoch + 1}/{epochs}, "
                         f"Reward: {episode_reward[-1]:.4f}, "
                         f"Sharpe Ratio: {episode_sharpe_ratio[-1]:.4f}, "
                         f"Epoch Time: {epoch_time:.2f}s, "
                         f"Total Time: {total_time:.2f}s")

    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time:.2f} seconds")

    return td3, episode_reward, episode_sharpe_ratio, training_times, all_actions


def load_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df = add_technical_indicators(df)
    return df

def evaluate_model_on_test_data(model, test_data):
    test_rewards = []
    state = test_data[0]

    for t in range(1, len(test_data)):
        action = model.select_action(state)
        next_state = test_data[t]
        reward = calculate_reward(action, next_state[3], state[3])
        test_rewards.append(reward)
        state = next_state

    sharpe_ratio = calculate_sharpe_ratio(np.array(test_rewards))
    logging.info(f"Test Sharpe Ratio: {sharpe_ratio:.4f}")
    return test_rewards, sharpe_ratio

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2018-01-01"
    end_date = "2023-12-31"

    stock_data = load_stock_data(ticker, start_date, end_date)
    trained_model, rewards, sharpe_ratios, training_times, all_actions = train_td3_stock_prediction(stock_data.values)

    torch.save(trained_model.actor.state_dict(), f'td3_stock_prediction_model_{ticker}.pth')
    logging.info(f"Model training completed and saved for {ticker}.")

    plot_training_metrics(rewards, sharpe_ratios, training_times)


    test_start_date = "2024-01-01"
    test_end_date = "2024-04-01"
    test_data = load_stock_data(ticker, test_start_date, test_end_date)


    test_rewards, test_sharpe_ratio = evaluate_model_on_test_data(trained_model, test_data.values)

    logging.info(f"Test Evaluation Completed: Sharpe Ratio {test_sharpe_ratio}")
