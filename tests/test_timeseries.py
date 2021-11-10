import numpy as np

import lstm_autoencoder


def test_adding_timeseries():
    series_1 = lstm_autoencoder.Constant(1)
    series_2 = lstm_autoencoder.Constant(2)
    series_3 = lstm_autoencoder.Constant(-10)

    series_4 = series_1 + series_2 + series_3

    reference_serie = lstm_autoencoder.Constant(-7)

    timestamps = np.arange(0, 30, 0.1)

    assert np.array_equal(series_4(timestamps), reference_serie(timestamps))


def test_multiplying_timeseries():
    series_1 = lstm_autoencoder.Constant(-1)
    series_2 = lstm_autoencoder.Constant(2)
    series_3 = lstm_autoencoder.Constant(-10)

    series_4 = series_1 * series_2 * series_3

    reference_serie = lstm_autoencoder.Constant(20)

    timestamps = np.arange(0, 30, 0.1)

    assert np.array_equal(series_4(timestamps), reference_serie(timestamps))


def test_multiplying_and_add_timeseries():
    series_1 = lstm_autoencoder.Constant(-1)
    series_2 = lstm_autoencoder.Constant(2)
    series_3 = lstm_autoencoder.Constant(10)

    series_4 = series_1 + series_2 * series_3

    reference_serie = lstm_autoencoder.Constant(19)

    timestamps = np.arange(0, 30, 0.1)

    assert np.array_equal(series_4(timestamps), reference_serie(timestamps))
