from __future__ import print_function

import numpy as np
import os
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201

import Utilities.utility_img as img_reader

batch_size = 8  # orig paper trained all networks with batch_size=128
epochs = 100
data_augmentation = False
num_classes = -1
log_dir = './logs/'

os.environ['CUDA_VISIBLE_DEVICES']='1'


train_path = "C:\\Users\\18053690439\\Desktop\\CoughClassification\\Data\\continuous_wavelet_transform_lite\\continuous_wavelet_transform\\train\\"
test_path = "C:\\Users\\18053690439\\Desktop\\CoughClassification\\Data\\continuous_wavelet_transform_lite\\continuous_wavelet_transform\\validation\\"

# train_path = "C:\\Users\\admin\\Desktop\\Caugh\\continuous_wavelet_transform_lite\\continuous_wavelet_transform\\train\\"
# test_path = "C:\\Users\\admin\\Desktop\\Caugh\\continuous_wavelet_transform_lite\\continuous_wavelet_transform\\validation\\"


# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

max_img_per_folder = 99999

# Model name, depth and version
model_type = 'DenseNet121_Image'

#Read Data
x_train, y_train, pathes, num_classes = img_reader.read_img(train_path, max_img_per_folder)
IMAGES = img_reader.img_class(x_train, y_train, pathes)
x_test, y_test, pathes, _ = img_reader.read_img(test_path, max_img_per_folder)
IMAGES_TEST = img_reader.img_class(x_test, y_test, pathes)

# Input image dimensions.
input_shape = IMAGES.data.shape[1:]

# Normalize data.
IMAGES.data = IMAGES.data.astype('float32') / 255
IMAGES_TEST.data = IMAGES_TEST.data.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(IMAGES.data, axis=0)
    IMAGES.data -= x_train_mean
    IMAGES_TEST.data -= x_train_mean

# Convert class vectors to binary class matrices.
IMAGES.label = keras.utils.to_categorical(IMAGES.label, num_classes)
IMAGES_TEST.label = keras.utils.to_categorical(IMAGES_TEST.label, num_classes)

# Decrease the lr by epoches
def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-3
    if epoch > epochs * 0.9:
        lr *= 0.5e-3
    elif epoch > epochs * 0.8:
        lr *= 1e-3
    elif epoch > epochs * 0.6:
        lr *= 1e-2
    elif epoch > epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

model = DenseNet121(weights=None,include_top=True,input_shape=(224,224,3),classes=2)
# Changed to Adagrad?
sgd = keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
print(model_type)
print(IMAGES.data.shape[0], 'train samples')
print(IMAGES_TEST.data.shape[0], 'test samples')
print('x_train shape:', IMAGES.data.shape)
print('y_train shape:', IMAGES.label.shape)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'Models')
model_name = 'CoughClassification_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             mode='max',
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler, TensorBoard(log_dir=log_dir,write_images=False)]

# Run training, with or without data augmentation.

model.fit(IMAGES.data, IMAGES.label,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(IMAGES_TEST.data, IMAGES_TEST.label),
            shuffle=True,
            callbacks=callbacks)

# Score trained model.

MATRIX = np.zeros((num_classes, num_classes))

pre_ans = model.predict(IMAGES_TEST.data, batch_size=32)
total = len(pre_ans)
right_sum = 0.0
for idx,ele in enumerate(pre_ans):
    idx_predict = np.argmax(IMAGES_TEST.label[idx])
    idx_label = np.argmax(pre_ans[idx])
    if(idx_predict == idx_label):
        right_sum += 1
    MATRIX[idx_predict][idx_label] += 1

# np.savetxt(export_res.get_script_name() + '_Matrix.csv', MATRIX, delimiter=',')
print('Last Accuracy: %f' % (right_sum/total) )
print('Total time cost: %f ' % (time.time() - script_start_time) )
# print(IMAGES_TEST.path[0])
