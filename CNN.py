from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
import idx2numpy
import os
import numpy as np
import cv2

comnist_labels = []
for i in range(42):
    comnist_labels.append(i)
comnist_labels_char = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 1040, 1041, 1042, 1043, 1044, 1045,  1046, 1047, 1048,
                       1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064,
                       1065, 1066, 1067, 1068, 1069, 1070, 1071]

'''
#Запись цифр из датасета EMNIST
comnist_path = os.path.abspath(os.curdir) + '\\CoMNIST\\'
X_train = idx2numpy.convert_from_file(comnist_path + 'emnist-digits-train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file(comnist_path + 'emnist-digits-train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file(comnist_path + 'emnist-digits-test-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file(comnist_path + 'emnist-digits-test-labels-idx1-ubyte')
x_train = np.zeros(X_train.shape)
x_test = np.zeros(X_test.shape)
for i in range(len(X_train)):
    x_train[i] = X_train[i].T
for i in range(len(X_test)):
    x_test[i] = X_test[i].T
for i in range(len(x_train)):
    cv2.imwrite(comnist_path + 'Cyrillic\\' + str(y_train[i]) + '\\' + str(i) + '.png', x_train[i])
for i in range(len(x_test)):
    cv2.imwrite(comnist_path + 'Cyrillic\\' + str(y_test[i]) + '\\' + str(i) + '.png', x_test[i])
'''

comnist_path = os.path.abspath(os.curdir) + '\\CoMNIST\\Cyrillic\\'
# Изменение размера изображений букв из Comnist
'''
for i in range(10, len(comnist_labels) + 1):
    files = os.listdir(comnist_path + str(i) + '\\')
    for j in range(len(files)):
        tmp_img = cv2.imread(comnist_path + str(i) + '\\' + files[j], -1)
        img = tmp_img[:, :, 1:4]
        img[:, :, 1] = img[:, :, 2]
        img[:, :, 0] = img[:, :, 2]
        output_img = cv2.resize(img, (28, 28))
        cv2.imwrite(comnist_path + str(i) + '\\' + files[j], output_img)
'''

X_train = []
Y_train = []
for i in range(len(comnist_labels)):
    files = os.listdir(comnist_path + str(i) + '\\')
    for j in range(len(files)):
        img = cv2.imread(comnist_path + str(i) + '\\' + files[j])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_train.append(gray)
        Y_train.append(comnist_labels[i])
    print(i)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)
X_train = X_train[idx, ...]
Y_train = Y_train[idx, ...]
X_test = X_train[:X_train.shape[0] // 20]
Y_test = Y_train[:Y_train.shape[0] // 20]
X_train = X_train[X_train.shape[0] // 20:]
Y_train = Y_train[Y_train.shape[0] // 20:]
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
X_train = X_train.astype(np.float32)
X_train /= 255.0
X_test = X_test.astype(np.float32)
X_test /= 255.0
x_train_cat = keras.utils.to_categorical(Y_train, len(comnist_labels))
y_test_cat = keras.utils.to_categorical(Y_test, len(comnist_labels))
'''
baseModel = tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False,
input_tensor=tf.keras.Input(shape=(28, 28, 3)))
headModel = baseModel.output
headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(tf.config.CLASSES), activation="softmax")(headModel)
model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)
'''
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(comnist_labels), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.001)

model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)

model.save('comnist_letters.h5')