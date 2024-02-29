import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard

from fcn16 import FCN16
import math
from load_batchs import image_segmentation_generator, generate_enhance, load_inputdata

if __name__ == '__main__':
    # # 训练用GPU设备
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # # GPU设备的内存设置为动态增长模式
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # 训练集
    train_images_path = "dataset/images_prepped_train/"
    train_segs_path = "dataset/annotations_prepped_train/"

    val_images_path = "dataset/images_prepped_test/"
    val_segs_path = "dataset/annotations_prepped_test/"

    nClasses = 15
    input_height = 320
    input_width = 320
    batch_size = 10

    fcn16 = FCN16(nClasses, input_height, input_width)

    if os.path.exists('output/fcn16_model.h5'):
        fcn16.load_weights('output/fcn16_model.h5')
        print('load pretrained model weights successfully!')

    fcn16.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    enhance_fn = generate_enhance(nClasses)
    train_images, train_segments = load_inputdata(train_images_path, train_segs_path)
    train_gen = image_segmentation_generator(train_images, train_segments,
                                             batch_size, input_height, input_width,
                                             enhance_fn)

    valid_images, valid_segments = load_inputdata(val_images_path, val_segs_path)
    valid_gen = image_segmentation_generator(valid_images, valid_segments,
                                             batch_size, input_height, input_width,
                                             enhance_fn)

    checkpoint = ModelCheckpoint(
        filepath="output/fcn16_model.h5",
        monitor='acc',
        mode='auto',
        save_weights_only=True,
        save_best_only=True)


    tensorboard = TensorBoard(log_dir='output/log_fcn16_model')

    fcn16.fit(
        x=train_gen,
        steps_per_epoch=math.ceil(len(train_images) / batch_size),  # 每轮批次数量
        epochs=10,
        shuffle=True,
        callbacks=[checkpoint, tensorboard])

