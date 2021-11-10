from dataclasses import dataclass
import math
import random
from typing import List

import numpy as np
import torch


run_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Encoder(torch.nn.Module):
    def __init__(self, num_timeseries: int, hidden_size: int, num_layers: int, dropout: float):
        super(Encoder, self).__init__()

        self._encoder = torch.nn.LSTM(num_timeseries, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        _, (hn, cn) = self._encoder(x)
        return hn, cn


class Decoder(torch.nn.Module):
    def __init__(self, num_timeseries: int, hidden_size: int, num_layers: int, dropout: float):
        super(Decoder, self).__init__()

        self._decoder = torch.nn.LSTM(num_timeseries, hidden_size, num_layers, dropout=dropout)
        self._decoder_output_mapping = torch.nn.Linear(hidden_size, num_timeseries)

    def forward(self, x, hn, cn):
        output, (hn, cn) = self._decoder(x, (hn, cn))
        return self._decoder_output_mapping(output), hn, cn


@dataclass
class TrainingParameters:
    epochs: int
    learning_rate: float
    minibatch_size: int
    shuffle: bool
    teacher_forcing_ratio: float


@dataclass
class Prediction:
    input: np.ndarray
    reconstruction: np.ndarray
    window_start: int


class TimeseriesWindowDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data: np.ndarray, window_size: int, time_between_windows: int):
        self._data = data
        self._window_size = window_size
        self._time_between_windows = time_between_windows
        self._max_window_idx = math.floor((data.shape[0] - self._window_size) / self._time_between_windows)

    def __len__(self) -> int:
        return self._max_window_idx

    def __getitem__(self, idx: int):
        start_data_index = idx * self._time_between_windows
        end_data_index = start_data_index + self._window_size

        window_data = self._data[start_data_index:end_data_index, :]
        return torch.from_numpy(window_data).float(), start_data_index


class Autoencoder:
    def __init__(
        self,
        num_timeseries: int,
        hidden_size: int,
        num_layers: int,
        encoder_dropout: float,
        decoder_dropout: float,
        window_size: int,
        time_between_windows: int,
    ):
        self._encoder = Encoder(num_timeseries, hidden_size, num_layers, encoder_dropout).to(run_device)
        self._decoder = Decoder(num_timeseries, hidden_size, num_layers, decoder_dropout).to(run_device)

        self._num_timeseries = num_timeseries
        self._window_size = window_size
        self._time_between_windows = time_between_windows

    def train(self, data: np.ndarray, training_parameters: TrainingParameters) -> None:
        dataloader = torch.utils.data.DataLoader(
            TimeseriesWindowDataset(data, self._window_size, self._time_between_windows),
            batch_size=training_parameters.minibatch_size,
            shuffle=training_parameters.shuffle,
        )

        encoder_optimizer = torch.optim.Adam(self._encoder.parameters(), lr=training_parameters.learning_rate)
        decoder_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=training_parameters.learning_rate)

        self._encoder.train()
        self._decoder.train()

        for epoch in range(training_parameters.epochs):
            for raw_minibatch_data, _ in dataloader:
                # Need to have dimensions in order [window_size, batch_size, features]
                minibatch_data = raw_minibatch_data.permute(1, 0, 2).to(run_device)
                assert minibatch_data.shape[0] == self._window_size
                assert minibatch_data.shape[2] == self._num_timeseries

                use_teacher_forcing = True if random.random() < training_parameters.teacher_forcing_ratio else False

                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()

                encoder_last_hidden_state, encoder_last_cell_state = self._encoder(minibatch_data)

                # Initialize network
                prev_hn = encoder_last_hidden_state
                prev_cn = encoder_last_cell_state
                prev_step_output = minibatch_data[-1, :, :].unsqueeze(0)

                collected_reconstruction = []

                for step_ii in range(self._window_size):
                    # Note that first reconstruction's input is the reconstructed value itself!
                    prev_step_output, prev_hn, prev_cn = self._decoder(prev_step_output.detach(), prev_hn, prev_cn)

                    # Need to store model outputs to compute reconstruction loss
                    collected_reconstruction.append(prev_step_output)

                    if use_teacher_forcing:
                        data_ind = self._window_size - 1 - step_ii
                        prev_step_output = minibatch_data[
                            data_ind : data_ind + 1, :, :  # noqa: E203
                        ]  # TODO: or just -1 and unsqueeze?

                # Reconstruction is generated from end to start
                # so the reconstruction must be reversed before
                # computing reconstruction loss
                collected_reconstruction.reverse()
                reconstruction = torch.vstack(collected_reconstruction).float()

                loss = (minibatch_data - reconstruction).pow(2).mean()
                loss.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()

            if epoch % 50 == 0 or epoch == training_parameters.epochs - 1:
                print("epoch: {} loss {}".format(epoch, loss.detach().item()))

    @torch.no_grad()
    def predict(self, data: np.ndarray, minibatch_size=50) -> List[Prediction]:
        dataloader = torch.utils.data.DataLoader(
            TimeseriesWindowDataset(data, self._window_size, self._time_between_windows),
            batch_size=minibatch_size,
            shuffle=False,
        )

        results = []

        self._encoder.eval()
        self._decoder.eval()

        for raw_minibatch_data, window_start in dataloader:
            # Need to have dimensions in order [window_size, batch_size, features]
            minibatch_data = raw_minibatch_data.permute(1, 0, 2).to(run_device)
            encoder_last_hidden_state, encoder_last_cell_state = self._encoder(minibatch_data)

            # Initialize network
            prev_hn = encoder_last_hidden_state
            prev_cn = encoder_last_cell_state
            prev_step_output = minibatch_data[-1, :, :].unsqueeze(0)

            collected_reconstruction = []

            for _ in range(self._window_size):
                # Note that first reconstruction's input is the reconstructed value itself!
                prev_step_output, prev_hn, prev_cn = self._decoder(prev_step_output.detach(), prev_hn, prev_cn)

                # Need to store model outputs to output reconstruction
                collected_reconstruction.append(prev_step_output)

            # Reconstruction is generated from end to start
            # so must reverse the reconstruction
            collected_reconstruction.reverse()
            reconstruction_np = torch.vstack(collected_reconstruction).detach().numpy()
            input_np = minibatch_data.detach().cpu().numpy()
            window_start_np = window_start.detach().cpu().numpy()

            assert reconstruction_np.shape == input_np.shape

            for sample_ii in range(input_np.shape[1]):
                results.append(
                    Prediction(
                        input=input_np[:, sample_ii, :],
                        reconstruction=reconstruction_np[:, sample_ii, :],
                        window_start=window_start_np[sample_ii],
                    )
                )

        return results
