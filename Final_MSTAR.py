# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:16:38 2024

@author: vivek
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array 
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
import seaborn as sns

train_data_dir = r"C:\Users\vivek\OneDrive\Documents\Msc_SAR_dataset\3classes_images\30_DEG"
validation_data_dir = r"C:\Users\vivek\OneDrive\Documents\Msc_SAR_dataset\3classes_images\45_DEG"

img_height, img_width = 128, 128
epochs = 50
batch_size = 32

def pref(image):
  return cv2.bilateralFilter(image, 60,100,100)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=10.,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=pref,validation_split = 0.2)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=pref,validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',subset='training')

validation_generator = test_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',subset = 'validation')

nb_train_samples = 892
nb_validation_samples = 222

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(128,128, 3)))
model.add(Convolution2D(32, 1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Convolution2D(64, 1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Convolution2D(128,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(256,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Convolution2D(512,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Convolution2D(512,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1028, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

checkpoint = ModelCheckpoint(r"C:\Users\vivek\OneDrive\Documents\Msc_SAR_dataset\Created_models/new_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
hist1 = model.fit(train_generator,steps_per_epoch=nb_train_samples// batch_size ,epochs = epochs, validation_data=validation_generator, validation_steps=nb_validation_samples// batch_size,workers = 4,callbacks=[checkpoint,early])

plt.plot(hist1.history["accuracy"])
plt.plot(hist1.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.show()

plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("Epoch")
plt.legend(["loss","validation loss"])
plt.show()

model_saved = load_model(r"C:\Users\vivek\OneDrive\Documents\Msc_SAR_dataset\Created_models/new_model.h5")

validation_generator1 = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode = 'rgb',
    class_mode='categorical')

test_datagen = ImageDataGenerator(
    rescale=1./255)
    #preprocessing_function=dims)

predictions = model_saved.predict(validation_generator1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator1.classes

cm = confusion_matrix(true_classes, predicted_classes)
print(cm)
class_labels = ['2S1', 'BRDM2','ZSU_23_4']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

cls_report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(cls_report)



layer = vgg_model.layers #Conv layers at 1, 3, 6, 8, 11, 13, 15
filters, biases= vgg_model.layers[0].get_weights()
print(layer[0].name, filters.shape)


fig1=plt.figure(figsize=(12,12))
columns = 8
rows = 4
n_filters = columns * rows
for i in range(1, n_filters +1):
    f = filters[:, :, :, i-1]
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray') 
plt.show() 


from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array 
import cv2

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(128,128, 3)))
model.add(Convolution2D(32, 1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64, 1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(128,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(256,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(512,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(512,1,1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

print(model.summary())  

layer = model.layers #Conv layers at 1, 3, 6, 8, 11, 13, 15
filters, biases= model.layers[1].get_weights()
print(layer[1].name, filters.shape)


fig1=plt.figure(figsize=(12,12))
columns = 8
rows = 4
n_filters = columns * rows
for i in range(1, n_filters +1):
    f = filters[:, :, :, i-1]
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray') 
plt.show() 


conv_layer_index = [1, 3, 5, 7, 9,11]  #TO define a shorter model
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

img = load_img(r"C:\Users\vivek\OneDrive\Documents\Msc_SAR_dataset\MSTAR-10-Classes\test\2S1\HB14973.jpeg", target_size=(128,128)) #VGG user 224 as input

img = img_to_array(img)
img = cv2.bilateralFilter(img, 60,100,100)
img = np.expand_dims(img, axis=0)
   
predictions = model_short.predict(img)

columns = 8
rows = 4
for ftr in predictions:
    #pos = 1
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        #pos += 1
    plt.show()