from keras.applications import vgg16
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, add, Dropout, Reshape, Activation

def FCN32_helper(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    
    img_input = Input(shape=(input_height, input_width, 3))
    
    # 加载VGG16预训练模型作为backbone
    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input,
        pooling=None,
        classes=1000
    )
    assert isinstance(model, Model)
    
    o = Conv2D(filters=4096, kernel_size=7, padding="same", activation="relu", name="fc6")(model.output)
    o = Dropout(0.5)(o)
    o = Conv2D(filters=4096, kernel_size=1, padding="same", activation="relu", name="fc7")(o)
    o = Dropout(0.5)(o)
    o = Conv2D(filters=nClasses, kernel_size=1, padding="same", activation="relu", kernel_initializer="he_normal", name="score_fr")(o)
    
    fcn32 = Model(inputs=img_input, outputs=o)
    
    return fcn32

def FCN32(nClasses, input_height, input_width):
    fcn32 = FCN32_helper(nClasses, input_height, input_width)
    
    Up = Conv2DTranspose(filters=nClasses, kernel_size=32, strides=32, padding="valid", activation=None, name="upsample")(fcn32.output)
    
    Up = Reshape((-1, nClasses))(Up)
    Up = Activation("softmax")(Up)
    
    mymodel = Model(inputs=fcn32.input, outputs=Up)
    
    return mymodel


if __name__ == "__main__":
    m = FCN32(15, 320, 320)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file="model_fcn32.png")
    m.summary()
 