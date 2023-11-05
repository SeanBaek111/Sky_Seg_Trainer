import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(DisplayCallback, self).__init__()
        self.train_accs = []
        self.val_accs = []
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
       
        # Add current epoch's accuracy and loss to the lists
        self.train_accs.append(logs.get('accuracy'))
        self.val_accs.append(logs.get('val_accuracy'))
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Plot training & validation accuracy values
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(epoch+1), self.train_accs)
        plt.plot(np.arange(epoch+1), self.val_accs)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(epoch+1), self.train_losses)
        plt.plot(np.arange(epoch+1), self.val_losses)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        plt.show()

        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))