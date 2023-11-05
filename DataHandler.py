import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataHandler:
    # Setting default class variables
    IMG_SIZE = 256
    N_CLASSES = 151
    BATCH_SIZE = 1
    SEED = 42
    DATASET_PATH = "ADEChallengeData2016/images/"
    TRAINING_DATA = "training/"
    VAL_DATA = "validation/"

    def __init__(self, root_path, img_size=None, batch_size=None, seed=None):
        # If parameters are provided during instantiation, override the class variables
        if img_size is not None:
            self.IMG_SIZE = img_size
        if batch_size is not None:
            self.BATCH_SIZE = batch_size
        if seed is not None:
            self.SEED = seed
            tf.random.set_seed(self.SEED)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_root_path = os.path.join(base_dir, root_path)
        self.DATASET_PATH = os.path.join(base_root_path,self.DATASET_PATH)
        self.TRAINSET_SIZE = len(glob(os.path.join(self.DATASET_PATH, self.TRAINING_DATA, "*.jpg")))
        self.VALSET_SIZE = len(glob(os.path.join(self.DATASET_PATH, self.VAL_DATA, "*.jpg")))

     
    def parse_image(img_path: str) -> dict:
        """Load an image and its annotation (mask) and returning
        a dictionary.
    
        Parameters
        ----------
        img_path : str
            Image (not the mask) location.
    
        Returns
        -------
        dict
            Dictionary mapping an image and its annotation.
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
    
        # For one Image path:
        # .../trainset/images/training/ADE_train_00000001.jpg
        # Its corresponding annotation path is:
        # .../trainset/annotations/training/ADE_train_00000001.png
        mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
        mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)
        # In scene parsing, "not labeled" = 255
        # But it will mess up with our N_CLASS = 150
        # Since 255 means the 255th class
        # Which doesn't exist
        mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
        # Note that we have to convert the new value (0)
        # With the same dtype than the tensor itself
    
        return {'image': image, 'segmentation_mask': mask}
    
    def load_image(self, datapoint: dict, training=True) -> tuple:
        image = datapoint['image']
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def data_augmentation(self, image):
        image = tf.image.random_flip_left_right(image)
        # Additional data augmentation steps can be added here
        return image
    
    def prepare_dataset(self, subset='training') -> tf.data.Dataset:
        if subset == 'training':
            data_gen_args = dict(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            data_gen = ImageDataGenerator(**data_gen_args)
        else:
            data_gen = ImageDataGenerator() 
            
        images_dir = os.path.join(self.DATASET_PATH, subset, 'images')
        masks_dir = os.path.join(self.DATASET_PATH, subset, 'annotations')
    
        image_file_paths = glob(os.path.join(images_dir, '*.jpg'))
        mask_file_paths = glob(os.path.join(masks_dir, '*.png'))
    
        # Create tuples of image and mask paths
        dataset = tf.data.Dataset.from_tensor_slices((image_file_paths, mask_file_paths))
    
        # Parse images and masks
        dataset = dataset.map(lambda img, mask: self.parse_image(img, mask))
    
        # Apply data augmentation here if necessary
        dataset = dataset.map(lambda x: (self.data_augmentation(x['image']), x['segmentation_mask']))
      
        dataset = dataset.map(lambda x, y: ({"image": x}, y))
    
        # Batching and prefetching for performance optimization
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
        return dataset

 
