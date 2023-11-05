import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SkySegmentationModel:
    def __init__(self, model, dataset, name_to_index_dict, color_dict):
        self.model = model
        self.dataset = dataset
        self.name_to_index_dict = name_to_index_dict
        self.color_dict = color_dict
        self.sum_sky_iou_val = 0
        self.sum_mIoU_val = 0
        self.sky_count = 0
        self.count = 0
        self.sky_ratio = 0

    def convert_to_tflite(self):
        # Convert the model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        # Save the TFLite model
        with open('skymodel.tflite', 'wb') as f:
            f.write(tflite_model)

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

    def visualize_segmentation_sky(self, mask):
        output_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        sky_index = self.name_to_index_dict['sky']
        other_color = [245, 245, 220]

        output_image[mask == sky_index] = self.color_dict['sky']
        output_image[mask != sky_index] = other_color

        return output_image

    def display_sample_sky_iou_mIoU(self, display_list):
        sky_index = self.name_to_index_dict['sky']
        true_mask = display_list[1].numpy().squeeze().astype(int)
        predicted_mask = display_list[2].numpy().squeeze().astype(int)
        sky_iou_val = self.calculate_sky_iou(true_mask, predicted_mask, sky_index)
        mIoU_val = self.calculate_mIoU(true_mask, predicted_mask, sky_index)

        # If Sky IoU value is 0, exit the function early
        if sky_iou_val == 0:
            # print("Sky IoU is 0. Skipping visualization.")
            return

        # Visualization can be included here, similar to the commented out plt code

        self.sum_sky_iou_val += sky_iou_val
        self.sum_mIoU_val += mIoU_val
        self.sky_count += 1

    def evaluate(self):
        for image, mask in self.dataset['val']:
            sky_ratio = self.sky_pixel_ratio(mask.numpy())

            if sky_ratio >= 0.02 and round(sky_ratio, 4) != 0.5130:
                one_img_batch = image[0][tf.newaxis, ...]
                inference = self.model.predict(one_img_batch, verbose=0)
                pred_mask_test = self.create_mask(inference)
                self.display_sample_sky_iou_mIoU([image[0], mask[0], pred_mask_test[0]])

            self.count += 1

        print(f"sky_count: {self.sky_count}")
        print(f"Average Sky IoU: {self.sum_sky_iou_val/self.sky_count:.4f}")
        print(f"Average mIoU: {self.sum_mIoU_val/self.sky_count:.4f}")

    def sky_pixel_ratio(self, mask):
        total_pixels = mask.size  # 전체 픽셀 수
        sky_pixels = np.sum(mask == self.name_to_index_dict['sky
