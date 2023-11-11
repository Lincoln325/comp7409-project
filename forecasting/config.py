class Config:
    EPOCHS = 100  # maximum training epochs
    LOOK_BACK = 120
    
    """
    Open,High,Low,Close,Volume
    """
    FEATURES = ["Close"]

    LR = 0.01