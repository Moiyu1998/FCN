from keras import Model
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, Concatenate, Conv2DTranspose

def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    
    img_input = Input(shape=(input_height, input_width, 3))
    
    # Encoder
    vgg_model = vgg16.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=img_input,
    )
    assert isinstance(vgg_model, Model)
    
    o = vgg_model.get_layer("block5_conv3").output
    o = Conv2D(1024, 3, padding="same", activation="relu")(o)
    o = Conv2D(1024, 3, padding="same", activation="relu")(o)
    o = BatchNormalization()(o)
    # Decoder block1
    o = Conv2DTranspose(512, 2, strides=2, padding="valid")(o)
    # skip connection
    o = Concatenate(axis=-1)([vgg_model.get_layer("block4_conv3").output, o])
    o = Conv2D(512, 3, padding="same", activation="relu")(o)
    o = Conv2D(512, 3, padding="same", activation="relu")(o)
    o = BatchNormalization()(o)
    
    # Decoder block2
    o = Conv2DTranspose(256, 2, strides=2, padding="valid")(o)
    # skip connection
    o = Concatenate(axis=-1)([vgg_model.get_layer("block3_conv3").output, o])
    o = Conv2D(256, 3, padding="same", activation="relu")(o)
    o = Conv2D(256, 3, padding="same", activation="relu")(o)
    o = BatchNormalization()(o)
    
    # Decoder block3
    o = Conv2DTranspose(128, 2, strides=2, padding="valid")(o)
    # skip connection
    o = Concatenate(axis=-1)([vgg_model.get_layer("block2_conv2").output, o])
    o = Conv2D(128, 3, padding="same", activation="relu")(o)
    o = Conv2D(128, 3, padding="same", activation="relu")(o)
    o = BatchNormalization()(o)
    
    # Decoder block4
    o = Conv2DTranspose(64, 2, strides=2, padding="valid")(o)
    # skip connection
    o = Concatenate(axis=-1)([vgg_model.get_layer("block1_conv2").output, o])
    o = Conv2D(64, 3, padding="same", activation="relu")(o)
    o = Conv2D(64, 3, padding="same", activation="relu")(o)
    o = BatchNormalization()(o)
    
    # segmentation mask
    o = Conv2D(nClasses, 1, padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)
    o = Reshape((-1, nClasses))(o)
    o = Activation("softmax")(o)
    model = Model(inputs=img_input, outputs=o)
    return model
    
    
if __name__ == "__main__":
    m = UNet(15, 320, 320)
    print(m.get_weights()[2])
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file="model_unet.png")
    m.summary()
    