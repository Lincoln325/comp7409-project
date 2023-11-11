class Config:
    EPOCHS = 100  # maximum training epochs
    LOOK_BACK = 120 # lookback period
    FEATURES = ["Close"] # features to use in training, available features include ["Open", "High" ,"Low", "Close", "Volume"]
    LR = 0.01 # learning rate