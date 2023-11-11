# comp7409-project

## Getting Started

### Prerequisites

Install dependencies
```sh
pip install -r requirements.txt
```

## Usage
1. Download data
    ```sh
    cd data
    python main.py
    cd ..
    ```

2. Modify config.py
    ```python
    class Config:
        EPOCHS = 100  # maximum training epochs
        LOOK_BACK = 120 # lookback period
        FEATURES = ["Close"] # features to use in training, available features include ["Open", "High" ,"Low", "Close", "Volume"]
        LR = 0.01 # learning rate
    ```
3. Train the model
    ```sh
    python train.py
    ```

## Directory Layout
    .
    ├── README.md                   
    ├── config.py                       # Configuration
    ├── data                            # Data folder
    │   ├── main.py                     # Data downloading script
    │   └── ticker.txt                  # Selected stocks
    ├── datasets.py                     # Dataset preprocessing and preparation script
    ├── example.ipynb                   # Open source example for stock forecasting
    ├── model.py                        # Model
    ├── runs                            # Results folder
    ├── train.py                        # Main training script
    └── utlis.py                        # Utilities
