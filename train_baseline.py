import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img

root = '/Users/jason/Downloads/small_images/'

train_datagen = ImageDataGenerator()
train_dataset = train_datagen.flow_from_directory(root+'train', batch_size=64, target_size=(64,64), class_mode='binary', color_mode='grayscale')

val_datagen = ImageDataGenerator()
val_dataset = val_datagen.flow_from_directory(root+'val', batch_size=8, target_size=(64,64), class_mode='binary', color_mode='grayscale')

test_datagen = ImageDataGenerator()
test_dataset = test_datagen.flow_from_directory(root+'test', batch_size=64, target_size=(64,64), class_mode='binary', color_mode='grayscale')

# each iter is (batch_size, 64, 64, 1)
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(64,64,1)))
cnn.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='sgd', loss='binary_crossentropy', metrics = ['accuracy'])
cnn.summary()

train_size = 5216
val_size = 16
test_size = 624
model = cnn.fit_generator(train_dataset, steps_per_epoch=train_size//64, epochs=1, validation_data=val_dataset, validation_steps=624)
acc = cnn.evaluate_generator(test_dataset,steps=624)
print(acc)
