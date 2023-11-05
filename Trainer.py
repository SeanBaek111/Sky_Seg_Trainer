import math
import os
from glob import glob
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from DisplayCallback import DisplayCallback  # The display callback for visualization

class Trainer:
    def __init__(self, model, dataset, batch_size=4, learning_rate=1e-3, weight_decay=2e-5, epochs=500):
        self.model = model
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
        # 체크포인트 경로 설정 및 생성
        checkpoint_dir = "checkpoints/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_filepath = os.path.join(checkpoint_dir, f"weights_{current_time}.h5")

        # ModelCheckpoint 콜백
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        # 학습률 스케줄러 콜백
        lrate_scheduler = LearningRateScheduler(self._step_decay)
        display_callback = DisplayCallback()
        return [checkpoint_callback, lrate_scheduler, display_callback]

    def _step_decay(self, epoch):
        initial_lrate = self.learning_rate
        drop = 0.5
        epochs_drop = 10.0  # 10 에폭마다 학습률을 절반으로 줄임
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def train(self):
        # 모델 컴파일
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])

        # 최신 체크포인트 로드
        all_h5_files = glob('checkpoints/*.h5')
        if all_h5_files:
            latest_h5_file = max(all_h5_files, key=os.path.getctime)
            print("Loading weights from:", latest_h5_file)
            self.model.load_weights(latest_h5_file)

        # 훈련 파라미터 설정
        steps_per_epoch = len(self.dataset['train']) // self.batch_size 
        validation_steps = len(self.dataset['val']) // self.batch_size

        print(self.dataset)
        # 모델 훈련
        return self.model.fit(
            self.dataset['train'], 
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.dataset['val'], 
            callbacks=self.callbacks
        )
        
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
 
