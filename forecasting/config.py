class Config:
    EPOCHS = 100  # maximum training epochs
    LOOK_BACK = 120 # lookback period
    MODEL = "V2" # model version, there are three versions
    FEATURES = ["Open", "High" ,"Low", "Close"] # features to use in training, available features include ["Open", "High" ,"Low", "Close", "Volume"]
    LR = 0.01 # learning rate