import torch
import torch.nn as nn

class AirQualityLSTM(nn.Module):

    def __init__(self,
                 input_size:  int,
                 hidden_size: int,
                 num_layers:  int,
                 output_size: int,
                 dropout:     float = 0.2):
        super(AirQualityLSTM, self).__init__()

        # LSTM layer
        # input_size  = number of features per time step
        # hidden_size = size of the LSTM's memory (hyperparameter)
        # num_layers  = how many LSTM layers stacked on top of each other
        # batch_first = True means input shape is (batch, time, features)
        # dropout     = applied between LSTM layers (only if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

        # Dropout applied after LSTM before FC layers
        self.dropout = nn.Dropout(dropout)

        # Fully connected output head
        # Maps LSTM output → 24 predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),  # compress 128 → 64
            nn.ReLU(),                    # activation: zero-out negatives
            nn.Dropout(dropout),          # regularize again
            nn.Linear(64, output_size)   # 64 → 24 predictions
        )

    def forward(self, x):
        # x shape: (batch_size, lookback=48, input_size)

        # Pass sequence through LSTM
        # lstm_out: output at every time step, shape (batch, 48, hidden)
        # We don't need the hidden/cell state here, so we use _
        lstm_out, _ = self.lstm(x)

        # Take ONLY the last time step
        # lstm_out[:, -1, :] means:
        #   : = all batches
        #   -1 = last time step (hour 48)
        #   : = all hidden units
        last_step = lstm_out[:, -1, :]

        # Apply dropout for regularization
        last_step = self.dropout(last_step)

        # Pass through FC layers to get 24 predictions
        output = self.fc(last_step)

        return output    # shape: (batch_size, 24)
