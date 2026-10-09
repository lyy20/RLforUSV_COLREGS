import torch
import torch.nn as nn

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ICM, self).__init__()
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        # 前向模型
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 反向模型
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, next_state, action):
        # 编码状态
        state_encoded = self.encoder(state)
        next_state_encoded = self.encoder(next_state)

        # 前向模型预测
        forward_input = torch.cat([state_encoded, action], dim=-1)
        predicted_next_state_encoded = self.forward_model(forward_input)

        # 反向模型预测
        inverse_input = torch.cat([state_encoded, next_state_encoded], dim=-1)
        predicted_action = self.inverse_model(inverse_input)

        return predicted_next_state_encoded, predicted_action