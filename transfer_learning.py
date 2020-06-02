
import numpy as np
#import tensorflow as tf
#from tensorflow import keras

import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img

from vis.visualization import visualize_saliency
from matplotlib import pyplot as plt
from PIL import Image

root = '/Users/jason/Downloads/small_images/'

train_datagen = ImageDataGenerator()
train_dataset = train_datagen.flow_from_directory(root+'train', batch_size=64, target_size=(64,64), class_mode='binary', color_mode='rgb')

val_datagen = ImageDataGenerator()
val_dataset = val_datagen.flow_from_directory(root+'val', batch_size=8, target_size=(64,64), class_mode='binary', color_mode='rgb')

test_datagen = ImageDataGenerator()
test_dataset = test_datagen.flow_from_directory(root+'test', batch_size=64, target_size=(64,64), class_mode='binary', color_mode='rgb')

trained_model = keras.applications.VGG16(weights='imagenet', input_shape=(64, 64, 3), include_top=False)

trained_model.trainable = False

cnn = Sequential([trained_model])
cnn.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Flatten())
cnn.add(Dense(units=64, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
cnn.summary()

train_size = 5216
val_size = 16
test_size = 624
model = cnn.fit_generator(train_dataset, steps_per_epoch=train_size//64, epochs=1, validation_data=val_dataset, validation_steps=624)
acc = cnn.evaluate_generator(test_dataset,steps=624)
print(acc)

from torchvision import transforms, utils

seed_img_path = '/Users/jason/Downloads/chest_xray/test/PNEUMONIA/person112_bacteria_538.jpeg'
seed_img = Image.open(seed_img_path)
seed_img = transforms.functional.resize(seed_img, (64, 64))
new_img = Image.new("RGB", seed_img.size)
new_img.paste(seed_img)
new_img = np.array(new_img)
print(new_img.shape)
saliency_img = visualize_saliency(cnn, layer_idx=-1, filter_indices=0, seed_input=new_img)
plt.imshow(saliency_img)
plt.show()
