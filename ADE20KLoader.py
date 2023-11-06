import os
import tensorflow as tf
import numpy as np
from glob import glob
import csv

class ADE20KLoader:
    def __init__(self, dataset_path, label_file, colormap_file, img_size=256, batch_size=32, seed=42):
        self.seed = seed
        self.img_size = img_size
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.autotune = tf.data.experimental.AUTOTUNE

        self.trainset_size = len(glob(os.path.join(dataset_path, "training/*.jpg")))
        self.valset_size = len(glob(os.path.join(dataset_path, "validation/*.jpg")))

        self.train_dataset = self._create_dataset("training")
        self.val_dataset = self._create_dataset("validation")

        self.class_names, self.color_dict = self._load_class_info(label_file, colormap_file)

    def _create_dataset(self, subset):
        files = tf.data.Dataset.list_files(os.path.join(self.dataset_path, subset, "*.jpg"), seed=self.seed)
        files = files.shard(num_shards=50, index=0)
        dataset = files.map(self._parse_image)
        return dataset

    def _parse_image(self, img_path: str) -> dict:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
        mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

        return {'image': image, 'segmentation_mask': mask}

    def _create_ade20k_label_colormap(self, file_path):
        colormap = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                colormap.append([int(c) for c in row])
        return np.asarray(colormap)

    def _load_class_info(self, class_info_path, colormap_path):
        ade20k_colormap = self._create_ade20k_label_colormap(colormap_path)
        with open(class_info_path, "r") as file:
            lines = file.readlines()[1:152]

        class_names = []
        color_dict = {}
        for idx, line in enumerate(lines):
            parts = line.split('\t')
            class_name = parts[4].strip()
            class_names.append(class_name)
            color_dict[class_name] = ade20k_colormap[idx]

        return class_names, color_dict

    @tf.function
    def _normalize(self, input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image, input_mask

    @tf.function
    def _load_image(self, datapoint: dict, training=True) -> tuple:
        input_image = tf.image.resize(datapoint['image'], (self.img_size, self.img_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.img_size, self.img_size))

        if training:
            if tf.random.uniform(()) > 0.5:
                input_image = tf.image.flip_left_right(input_image)
                input_mask = tf.image.flip_left_right(input_mask)
            input_image = tf.image.random_brightness(input_image, max_delta=0.3)
            input_image = tf.image.random_contrast(input_image, lower=0.8, upper=1.2)

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

    def _filter_label(self, image, label):
        label = tf.where(tf.equal(label, len(self.class_names)), tf.cast(tf.constant(0), label.dtype), label)
        return image, label

    def prepare_dataset(self):
        self.train_dataset = self.train_dataset.map(
            lambda x: self._load_image(x, training=True),
            num_parallel_calls=self.autotune)
        self.train_dataset = self.train_dataset.map(
            self._filter_label,
            num_parallel_calls=self.autotune)
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=1000, seed=self.seed).batch(self.batch_size).prefetch(self.autotune)

        self.val_dataset = self.val_dataset.map(
            lambda x: self._load_image(x, training=False)).map(
            self._filter_label).batch(self.batch_size).prefetch(self.autotune)
        dataset = {"train": self.train_dataset, "val": self.val_dataset}
       
        return dataset


