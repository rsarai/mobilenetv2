from train import train
import matplotlib.pyplot as plt


def plot_graphs(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def run():
    classes = 10
    size = 28
    batch = 96
    epochs = 300
    weights = False
    tclasses = 0
    history = train(batch, epochs, classes, size, weights, tclasses)
    return history


history = run()
plot_graphs(plot_graphs)
