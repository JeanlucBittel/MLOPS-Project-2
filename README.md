# DistilBERT base model (uncased) on paraphrase detection (MRPC)
This project finetunes the [DistilBERT base model](https://huggingface.co/distilbert-base-uncased) and runs it in a [Docker](https://www.docker.com/) container.

## Table of Contents
* [Technologies](#technologies)
* [Setup](#setup)
* [Usage](#usage)
    * [Docker](#docker)
    * [Local](#local)

## Technologies
Project is created with:
* Python 3.10
* PyTorch 2.1
* PyTorch Lightning 1.9
* Datasets 2.14
* Transformers 4.35
* Weights & Biases 0.16

## Setup
Clone the repository to your local machine and install the required depencies.

With pip:
```console
pip install -r requirements.txt
```

Or conda:
```console
conda create –f environment.yml
```

If you want to log your results with Weights & Biases, then create a `.env` file in the root directory and add the following line:

```console
WANDB_API_KEY=$YOUR_API_KEY
```

> Note: Replace `$YOUR_API_KEY` with your actual API key.

## Usage
### Docker
In order to run the script in a Docker container you need to first build an image with:

```console
docker build -t python-imagename .
```

If you don't want to clone the repository you can also directly create an image from the repository using the following command:

```console
docker build -t python-imagename https://github.com/JeanlucBittel/MLOPS-Project-2.git
```

Once the image has been build you start it with:

```console
docker run python-imagename
```

> Note: When running the script with Docker you can't add any additional options. If you wish to use any options see the [next chapter](#locally).

### Local
If you want to start a run without Docker you can use the following command:

```console
py main.py
```

Or to see additional options use:

```console
py main.py -h
```
