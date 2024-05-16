import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.predictor_input_size)
        self.lstm = nn.LSTM(
            input_size=config.predictor_input_size,
            hidden_size=config.predictor_hidden_size,
            num_layers=config.predictor_num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(config.predictor_hidden_size, config.jointer_hidden_size),
            nn.LayerNorm(config.jointer_hidden_size),
        )

    def forward(self, input_ids):
        input_embeds = self.embedding(input_ids)
        lstm_output, hidden_states = self.lstm(input_embeds)
        predictor_output = self.fc(lstm_output)
        return predictor_output, hidden_states


class Jointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_head = nn.Sequential(
            nn.LayerNorm(config.encoder_hidden_size),
            nn.Linear(config.encoder_hidden_size, config.jointer_hidden_size),
        )
        self.fc = nn.Sequential(
            nn.Linear(config.jointer_hidden_size, config.jointer_hidden_size),
            nn.LeakyReLU(0.2),
        )
        self.lm_head = nn.Linear(config.jointer_hidden_size, config.vocab_size)

    def forward(self, acoustic_out, predictor_out):
        acoustic_out = self.encoder_head(acoustic_out)
        acoustic_out = acoustic_out.unsqueeze(2)  # [B, T, 1, H]
        predictor_out = predictor_out.unsqueeze(1)  # [B, 1, U, H]
        joint_input = acoustic_out + predictor_out  # [B, T, U, H]
        joint_input = torch.tanh(joint_input)
        joint_output = self.fc(joint_input)
        joint_output = self.lm_head(joint_output)
        return joint_output
