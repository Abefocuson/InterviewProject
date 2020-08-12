from __future__ import print_function
import time
script_start_time = time.time()
import keras
from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
import Utilities.utility_img as img_reader
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
import ExportResult as export_res

timestr = "20200812_"

test_path = "C:\\Users\\admin\\Desktop\\Caugh\\audio\\audio\\test\\"

model_path = 'Jianlong_Image_Net_pretrain_Cough_DenseNet_Audio121v0_model.036.h5'

max_img_per_folder = 99999
num_classes = 2
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

n = 3

version = 0

# Computed depth from supplied model parameter n
depth = 121

# Model name, depth and version
model_type = 'DenseNet%dv%d' % (depth, version)

#Read Data
x_test, y_test, pathes, _ = img_reader.read_img(test_path, max_img_per_folder)
IMAGES_TEST = img_reader.img_class(x_test, y_test, pathes)

# Input image dimensions.
input_shape = IMAGES_TEST.data.shape[1:]

# Normalize data.
IMAGES_TEST.data = IMAGES_TEST.data.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(IMAGES_TEST.data, axis=0)
    IMAGES_TEST.data -= x_train_mean

# Convert class vectors to binary class matrices.
IMAGES_TEST.label = keras.utils.to_categorical(IMAGES_TEST.label, num_classes)

model = DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3))



top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))
top_model.summary()

model = Model(inputs=model.input, outputs=top_model(model.output))

model.load_weights(model_path)
model.summary()
print(model_type)
print(IMAGES_TEST.data.shape[0], 'test samples')
print('x_train shape:', IMAGES_TEST.data.shape)
print('y_train shape:', IMAGES_TEST.label.shape)

# Score trained model.

MATRIX = np.zeros((num_classes, num_classes))
logist_out = []
predict_label = []
pre_ans = model.predict(IMAGES_TEST.data, batch_size=32)
total = len(pre_ans)
right_sum = 0.0
for idx,ele in enumerate(pre_ans):
    idx_predict = np.argmax(IMAGES_TEST.label[idx])
    idx_label = np.argmax(pre_ans[idx])
    if(idx_predict == idx_label):
        right_sum += 1
    MATRIX[idx_predict][idx_label] += 1
    predict_label.append(idx_predict)

pre_ans = pre_ans.transpose()
export_res.export_xml_result(export_res.get_script_name() + timestr+'_Result.xls',
[IMAGES_TEST.path, np.argmax(IMAGES_TEST.label,axis=1),predict_label,pre_ans[0],pre_ans[1]],
['File_Path','Right_Label','Predict','L0','L1']
)

np.savetxt(export_res.get_script_name() + timestr+ '_Matrix.csv', MATRIX, delimiter=',')
print('Last Accuracy: %f' % (right_sum/total) )
print('Total time cost: ' + str(time.time() - script_start_time) )

