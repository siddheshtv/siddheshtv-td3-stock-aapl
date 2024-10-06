import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time
from scipy import stats
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
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
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.max_action = max_action
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=1024, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_metrics(returns):
    total_return = np.sum(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    sortino_ratio = calculate_sortino_ratio(returns)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    volatility = np.std(returns) * np.sqrt(252)
    win_rate = np.sum(returns > 0) / len(returns)
    max_drawdown = calculate_max_drawdown(returns)
    calmar_ratio = calculate_calmar_ratio(returns, max_drawdown)

    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "VaR (95%)": var_95,
        "CVaR (95%)": cvar_95,
        "Volatility": volatility,
        "Win Rate": win_rate,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio
    }

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

def calculate_max_drawdown(returns):
    cum_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    return np.max(drawdown)

def calculate_calmar_ratio(returns, max_drawdown):
    annual_return = np.mean(returns) * 252
    return annual_return / abs(max_drawdown)

def plot_training_metrics(rewards, sharpe_ratios, training_times):
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    axs[0].plot(rewards)
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')

    axs[1].plot(sharpe_ratios)
    axs[1].set_title('Episode Sharpe Ratios')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Sharpe Ratio')

    axs[2].plot(training_times)
    axs[2].set_title('Training Time per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_returns_distribution(returns):
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, kde=True)
    plt.title('Distribution of Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig('returns_distribution.png')
    plt.close()

def plot_cumulative_returns(returns):
    cum_returns = np.cumsum(returns)
    plt.figure(figsize=(10, 6))
    plt.plot(cum_returns)
    plt.title('Cumulative Returns')
    plt.xlabel('Trading Day')
    plt.ylabel('Cumulative Return')
    plt.savefig('cumulative_returns.png')
    plt.close()

def plot_drawdown(returns):
    cum_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown)
    plt.title('Drawdown')
    plt.xlabel('Trading Day')
    plt.ylabel('Drawdown')
    plt.savefig('drawdown.png')
    plt.close()

def plot_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Model Outputs')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def train_td3_stock_prediction(stock_data, epochs=100, batch_size=1024):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(stock_data)

    state_dim = scaled_data.shape[1]
    action_dim = 1
    max_action = 1

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
            reward = -abs(action - next_state[0])
            replay_buffer.add(state, action, reward, next_state)

            episode_return += reward
            returns.append(reward)
            epoch_actions.append(action)

            state = next_state

            if len(replay_buffer.storage) > batch_size:
                td3.train(replay_buffer, 10, batch_size)

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
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"

    stock_data = load_stock_data(ticker, start_date, end_date)
    trained_model, rewards, sharpe_ratios, training_times, all_actions = train_td3_stock_prediction(stock_data.values)


    torch.save(trained_model.actor.state_dict(), f'td3_stock_prediction_model_{ticker}.pth')
    logging.info(f"Model training completed and saved for {ticker}.")


    plot_training_metrics(rewards, sharpe_ratios, training_times)


    final_returns = np.array(rewards[-1])
    metrics = calculate_metrics(final_returns)

    print("\nFinal Performance Metrics:")
    print(tabulate([(k, v) for k, v in metrics.items()], headers=['Metric', 'Value'], tablefmt='pretty'))


    plot_returns_distribution(final_returns)


    plot_cumulative_returns(final_returns)


    plot_drawdown(final_returns)


    all_actions_array = np.array(all_actions)
    correlation_matrix = np.corrcoef(all_actions_array.T)
    plot_heatmap(correlation_matrix)


    print("\nAdditional Statistics:")
    print(f"Total Training Time: {sum(training_times):.2f} seconds")
    print(f"Average Time per Epoch: {np.mean(training_times):.2f} seconds")
    print(f"Convergence Speed: Reached 90% of max Sharpe ratio at epoch {np.argmax(np.array(sharpe_ratios) >= 0.9 * max(sharpe_ratios))}")
    print(f"Stability of Policy: Std Dev of Actions = {np.std(all_actions_array):.4f}")


    import resource
    print(f"Peak Memory Usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB")

    logging.info("All analysis completed and saved.")