class Config:
    """
    The time frame to be used.
    short - 15 days
    mid - 60 days
    long - 120 days

    example:
    ["long"]
    ["short", "long"]
    ["short", "mid", "long"]
    """

    WINDOW = "long"

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
    FEATURES = ["v"]

    LR = 0.01  # initial learning rate
    EPOCHS = 1000  # maximum training epochs
    BATCH_SIZE = 32  # training and validation batch size
