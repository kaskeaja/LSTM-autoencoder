[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# LSTM autoencoder #

Simple LSTM autoencoder implementation

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

Create virtual env and install pip packages
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Create and load data
```
./create_simple_data.sh
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
