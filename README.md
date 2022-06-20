[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# LSTM autoencoder #

LSTM sequence to sequence model anomaly detection based on <a href="https://arxiv.org/abs/1607.00148v2">LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection</a>

Checkout the [notebook](notebooks/run.ipynb) to see how the model works on synthetic data.

### How do I get set up? ###

Create virtual env and install pip packages
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


### Running tests ###

To run all checks write
```
tox
```

To check coding conventions
```
flake
black --diff
```

To check typing write
```
pytype lstm_autoencoder
pytype tests
```
