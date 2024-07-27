# CVNLP Image Classification Project

This project involves deep learning for image classification using the HAM10000 dataset. The script `cvnlp.py` trains and evaluates a model for binary classification of skin lesions.

## Prerequisites

- Docker installed on your system

## Preparation of the dataset

Ensure that the HAM10000 dataset is available in the 'cvnlp' project folder in the following structure:

```
HAM10000
├── train
│ ├── images
│ └── metadata.csv
└── test
├── images
└── metadata.csv
```


## Build and start the container

Create the docker image (execute this within the 'cvnlp' project folder on the first level)

```sh
docker build -t cvnlp-image .
```

Run the docker container (execute this within the 'cvnlp' project folder on the first level)

```sh
docker run -v $(pwd):/app -it cvnlp-image /bin/bash
```

```sh
python experiments.py
```