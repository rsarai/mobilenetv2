from train import train


def run():
    classes = 10
    size = 28
    batch = 96
    epochs = 300
    weights = False
    tclasses = 0
    train(batch, epochs, classes, size, weights, tclasses)


run()
