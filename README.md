# comp7409-project

## Getting Started

### Prerequisites

Intall dependencies
```sh
pip install -r requirements
```

## Usage
1. Download data
    ```sh
    python data.py
    ```

2. Modify config.py
    ```python
    class Config:
    """
    The time frame to be used.
    short - 7 days
    mid - 15 days
    long - 30 days

    example:
    ["long"]
    ["short", "long"]
    ["short", "mid", "long"]
    """

    WINDOWS = ["long"]

    """
    The features to be used.
    o - open
    h - high
    l - low
    c - close
    v - volumn

    example:
    ["c"]
    ["o", "c"]
    ["o", "h", "l", "c", "v"]
    """
    FEATURES = ["o", "h", "l", "c", "v"]

    LR = 0.001  # initial learning rate
    EPOCHS = 100  # maximum training epochs
    BATCH_SIZE = 32  # training and validation batch size

    ```
3. Train the model
    ```sh
    python train.py
    ```


## Directory Layout

    .
    ├── config.py               # Training configuration
    ├── data.py                 # Download data
    ├── datasets.py             # Pytorch Datasets and Dataloaders
    ├── model.py                # Simple Pytorch time series model
    ├── train.py                # Main training scipt
    ├── utlis.py                # Utilities
    ├── trial.ipynb             # Notebook for quick trial and dirty code
    ├── requirements.txt        
    ├── README.md