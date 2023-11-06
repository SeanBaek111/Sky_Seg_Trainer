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
        print("sum_sky_iou_val", self.sum_sky_iou_val)
        print(f"sky_count: {self.sky_count}")
        print(f"Average Sky IoU: {self.sum_sky_iou_val/self.sky_count:.4f}")
        print(f"Average mIoU: {self.sum_mIoU_val/self.sky_count:.4f}")

    def create_mask(self, pred):
        return tf.argmax(pred, axis=-1) 

    def calculate_sky_iou(self, mask_true, mask_pred, class_id):
        intersection = np.sum((mask_true == class_id) & (mask_pred == class_id))
        union = np.sum((mask_true == class_id) | (mask_pred == class_id))
        return intersection / union if union != 0 else 0

    def calculate_mIoU(self, mask_true, mask_pred, class_id):
        # Sky IoU
        sky_intersection = np.sum((mask_true == class_id) & (mask_pred == class_id))
        sky_union = np.sum((mask_true == class_id) | (mask_pred == class_id))
        sky_iou = sky_intersection / sky_union if sky_union != 0 else 0
    
        # Non-Sky IoU
        non_sky_intersection = np.sum((mask_true != class_id) & (mask_pred != class_id))
        non_sky_union = np.sum((mask_true != class_id) | (mask_pred != class_id))
        non_sky_iou = non_sky_intersection / non_sky_union if non_sky_union != 0 else 0
    
        # mIoU
        mIoU = (sky_iou + non_sky_iou) / 2
        
        return mIoU

    def sky_pixel_ratio(self, mask):
        total_pixels = mask.size
        sky_pixels = np.sum(mask == self.name_to_index_dict['sky'])
        return sky_pixels / total_pixels if total_pixels > 0 else 0

    def display_sample_sky_iou_mIoU(self,display_list ):
        # global sum_sky_iou_val
        # global sum_mIoU_val
        # global sky_count
        # global count
        # global sky_ratio
        sky_index = self.name_to_index_dict['sky']
        true_mask = display_list[1].numpy().squeeze().astype(int)
        predicted_mask = display_list[2].numpy().squeeze().astype(int)
        sky_iou_val = self.calculate_sky_iou(true_mask, predicted_mask, sky_index)
        mIoU_val = self.calculate_mIoU(true_mask, predicted_mask, sky_index)
        
      #  If Sky IoU value is 0, exit the function early
        # if sky_iou_val == 0:
        #     # print("Sky IoU is 0. Skipping visualization.")
        #    return
      
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
            if i == 1 or i == 2:
                mask_to_display = img_to_display.numpy().squeeze().astype(int)
                display_img = self.visualize_segmentation_sky(mask_to_display)
                plt.imshow(display_img, interpolation='lanczos')
    
            else:
                plt.imshow(tf.keras.preprocessing.image.array_to_img(img_to_display))
            plt.axis('off')
        plt.show()
    
      #  print(f"sky_ratio: {sky_ratio:.4f}" )
        print(f"Sky IoU: {sky_iou_val:.4f}", end=" | ")
        sum_sky_iou_val += sky_iou_val
    
        print(f"mIoU: {mIoU_val:.4f}", end=" ")
        sum_mIoU_val += mIoU_val
    
        sky_count += 1
        print(sky_count, count)


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
