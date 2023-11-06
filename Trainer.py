import math
import os
from glob import glob
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from DisplayCallback import DisplayCallback  # The display callback for visualization

class Trainer:
    def __init__(self, model, data_loader, dataset, batch_size=4, learning_rate=1e-3, weight_decay=2e-5, epochs=500):
        self.model = model

        self.data_loader = data_loader
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimizer = self._get_optimizer()
        self.callbacks = self._init_callbacks()
        self.train_accs = []
        self.val_accs = []
        self.train_losses = []
        self.val_losses = []
        
    def _get_optimizer(self):
        return tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)

    def _init_callbacks(self):
        
        checkpoint_dir = "checkpoints/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_filepath = os.path.join(checkpoint_dir, f"weights_{current_time}.h5")

        # ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        for image, mask in self.dataset['train'].skip(1).take(1):
            sample_image, sample_mask = image, mask
 
        lrate_scheduler = LearningRateScheduler(self._step_decay)

        class_names = self.data_loader.class_names
        color_dict = self.data_loader.color_dict
        
        display_callback = DisplayCallback(self.model, sample_image, sample_mask, class_names, color_dict)
        return [checkpoint_callback, lrate_scheduler, display_callback]

    def _step_decay(self, epoch):
        initial_lrate = self.learning_rate
        drop = 0.5
        epochs_drop = 10.0  # decrease lr every 10 epochs
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def train(self): 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])

       
        all_h5_files = glob('checkpoints/*.h5')
        if all_h5_files:
            latest_h5_file = max(all_h5_files, key=os.path.getctime)
            print("Loading weights from:", latest_h5_file)
            self.model.load_weights(latest_h5_file)

        
        steps_per_epoch = len(self.dataset['train']) // self.batch_size 
        validation_steps = len(self.dataset['val']) // self.batch_size

        print(self.dataset)
        
        return self.model.fit(
            self.dataset['train'], 
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.dataset['val'], 
            callbacks=self.callbacks
        )

     
    