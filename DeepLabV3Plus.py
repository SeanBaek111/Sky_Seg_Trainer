import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DeepLabV3Plus:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = self.build_model()

    def convolution_block(self, block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x)

    def DilatedSpatialPyramidPooling(self, dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, num_filters=256, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
        )(x)

        out_1 = self.convolution_block(dspp_input, num_filters=256, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, num_filters=256, kernel_size=1)
        return output

    def build_model(self):
        model_input = keras.Input(shape=(self.image_size, self.image_size, 3), name='image')   
        resnet50 = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(
            size=(self.image_size // 4 // x.shape[1], self.image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x, num_filters=256)
        x = self.convolution_block(x, num_filters=256)
        x = layers.UpSampling2D(
            size=(self.image_size // x.shape[1], self.image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(self.num_classes, kernel_size=(1, 1), activation='softmax', padding="same")(x)

        return keras.Model(inputs=model_input, outputs=model_output)
