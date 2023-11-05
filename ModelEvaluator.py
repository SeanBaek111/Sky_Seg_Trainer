import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class ModelEvaluator:
    def __init__(self, model, dataset, name_to_index_dict, color_dict):
        self.model = model
        self.dataset = dataset
        self.name_to_index_dict = name_to_index_dict
        self.color_dict = color_dict

        self.sum_sky_iou_val = 0
        self.sum_mIoU_val = 0
        self.sky_count = 0
        self.count = 0

    def evaluate(self):
        for image, mask in self.dataset['val']:
            self.count += 1
            sky_ratio = self.sky_pixel_ratio(mask.numpy())
    
            if sky_ratio >= 0.02:  # Only consider masks with enough sky
                # Ensure the image has the shape (256, 256, 3)
                if image.ndim == 3 and image.shape[0] == 256 and image.shape[1] == 256 and image.shape[2] == 3:
                    # Add a batch dimension
                    one_img_batch = image[tf.newaxis, ...]
                elif image.ndim == 4 and image.shape[1] == 256 and image.shape[2] == 256 and image.shape[3] == 3:
                    # Image already has a batch dimension
                    one_img_batch = image
                else:
                    # Invalid shape, raise an error or continue with the next image
                    print(f"Invalid image shape: {image.shape}")
                    continue
    
                pred_mask = self.create_mask(self.model.predict(one_img_batch, verbose=False))
                self.display_sample_sky_iou_mIoU([image, mask, pred_mask])
                self.sky_count += 1

        print(f"sky_count: {self.sky_count}")
        print(f"Average Sky IoU: {self.sum_sky_iou_val/self.sky_count:.4f}")
        print(f"Average mIoU: {self.sum_mIoU_val/self.sky_count:.4f}")

    def create_mask(self, pred):
        return tf.argmax(pred, axis=-1)

    def calculate_sky_iou(self, mask_true, mask_pred):
        sky_index = self.name_to_index_dict['sky']
        intersection = np.sum((mask_true == sky_index) & (mask_pred == sky_index))
        union = np.sum((mask_true == sky_index) | (mask_pred == sky_index))
        return intersection / union if union != 0 else 0

    def calculate_mIoU(self, mask_true, mask_pred):
        sky_iou = self.calculate_sky_iou(mask_true, mask_pred)
        non_sky_iou = self.calculate_sky_iou(1 - mask_true, 1 - mask_pred)  # Assuming binary mask
        return (sky_iou + non_sky_iou) / 2

    def sky_pixel_ratio(self, mask):
        total_pixels = mask.size
        sky_pixels = np.sum(mask == self.name_to_index_dict['sky'])
        return sky_pixels / total_pixels if total_pixels > 0 else 0

    def display_sample_sky_iou_mIoU(self, display_list):
        true_mask = display_list[1].numpy().squeeze().astype(int)
        predicted_mask = display_list[2].numpy().squeeze().astype(int)
        sky_iou_val = self.calculate_sky_iou(true_mask, predicted_mask)
        mIoU_val = self.calculate_mIoU(true_mask, predicted_mask)

        # When preparing the image for display, remove the batch dimension
        if sky_iou_val > 0:
            for i in range(len(display_list)):
                # Squeeze out the batch dimension if it's there
                display_image = np.squeeze(display_list[i])
                
                # Check if the image has an acceptable shape for display
                if display_image.ndim == 3 and display_image.shape[-1] in {1, 3, 4}:
                    self.visualize_sample(display_image)
                else:
                    print(f"Invalid image shape for display: {display_image.shape}")
                    continue
                
                print(f"Sky IoU: {sky_iou_val:.4f} | mIoU: {mIoU_val:.4f}")
                self.sum_sky_iou_val += sky_iou_val

    def visualize_sample(self, display_list):
        plt.figure(figsize=(18, 18))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            
            img_to_display = display_list[i]
            if i > 0:  # For masks
                img_to_display = self.visualize_segmentation_sky(img_to_display.squeeze().astype(int))

            
            plt.imshow(img_to_display)
            plt.axis('off')
        plt.show()

    def visualize_segmentation_sky(self, mask):
        output_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        sky_index = self.name_to_index_dict['sky']
        other_color = [245, 245, 220]  # Beige
        
        output_image[mask == sky_index] = self.color_dict['sky']
        output_image[mask != sky_index] = other_color
                
        return output_image
