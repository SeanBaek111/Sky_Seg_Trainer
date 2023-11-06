import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_image, sample_mask, class_names, color_dict):
        super(DisplayCallback, self).__init__()
        self.model = model
        self.sample_image = sample_image
        self.sample_mask = sample_mask
       # self.class_names = class_names
        self.color_dict = color_dict
        self.name_to_index_dict = {name: index for index, name in enumerate(class_names, start=0)}
        self.train_accs = []
        self.val_accs = []
        self.train_losses = []
        self.val_losses = []
        

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        self.epoch = epoch
        self.show_predictions()
        
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
        
    def show_predictions(self, dataset=None, num=3):
        """Show a sample prediction.
    
        Parameters
        ----------
        dataset : [type], optional
            [Input dataset, by default None
        num : int, optional
            Number of sample to show, by default 1
        """
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                self.display_sample([image[0], true_mask, create_mask(pred_mask)])
        else:
            # The model is expecting a tensor of the size
            # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
            # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
            # and we want only 1 inference to be faster
            # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
            one_img_batch = self.sample_image[0][tf.newaxis, ...]
            # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
            inference = self.model.predict(one_img_batch)
            # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
            pred_mask = self.create_mask(inference)
            # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
            self.display_sample([self.sample_image[0], self.sample_mask[0],
                            pred_mask[0]])

        print ('\nSample Prediction after epoch {}\n'.format(self.epoch+1))
        
    def create_mask(self, pred_mask: tf.Tensor) -> tf.Tensor:
        """Return a filter mask with the top 1 predicitons
        only.
    
        Parameters
        ----------
        pred_mask : tf.Tensor
            A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
            N_CLASS values (vector) which represents the probability of the pixel
            being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
            has been predicted class 2 with a probability of 100%.
    
        Returns
        -------
        tf.Tensor
            A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
            for each pixels.
        """
        # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
        # 1 prediction for each class but we want the highest score only
        # so we use argmax
        pred_mask = tf.argmax(pred_mask, axis=-1)
        # pred_mask becomes [IMG_SIZE, IMG_SIZE]
        # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
        pred_mask = tf.expand_dims(pred_mask, axis=-1)
        return pred_mask

    def display_sample(self, display_list):
        """Show side-by-side an input image, the ground truth, and the prediction."""
        plt.figure(figsize=(18, 18))
        title = ['Input Image', 'True Mask', 'Predicted Mask']
    
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            
            # Ensure the image has 3 channels
            img_to_display = display_list[i]
            if len(img_to_display.shape) == 2:
                img_to_display = np.expand_dims(img_to_display, axis=-1)
            
            # For ground truth and predicted mask, use the colormap
            if i == 1 or i == 2:  # i==1 is True Mask, i==2 is Predicted Mask
                mask_to_display = img_to_display.numpy().squeeze().astype(int)
               # print(mask_to_display)
               # print(mask_to_display.min(), mask_to_display.max())
    
                
                display_img = self.visualize_segmentation(mask_to_display, self.color_dict, self.name_to_index_dict)
    
                plt.imshow(display_img, interpolation='lanczos')
    
            else:
                plt.imshow(tf.keras.preprocessing.image.array_to_img(img_to_display))
            plt.axis('off')
        plt.show()

    def visualize_segmentation(self, mask, color_dict, name_to_index_dict):
        output_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for name, idx in name_to_index_dict.items():
           # if name == 'sky':
           #     print(idx, name)
            output_image[mask == idx] = color_dict[name]
        return output_image
