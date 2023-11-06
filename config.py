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
