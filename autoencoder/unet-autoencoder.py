from keras_unet.models import custom_unet
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyart
import glob

input_shape = (256, 256, 3)
autoencoder = custom_unet(input_shape=input_shape, output_activation='sigmoid')
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

#train the model
train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        featurewise_std_normalization=True,
        featurewise_center=True)

x_train = train_datagen.flow_from_directory(
        '../data/images/',
        target_size=(256, 256),
        batch_size=10,
        class_mode='binary')
x_test = train_datagen.flow_from_directory(
        '../data/images/',
        target_size=(256, 256),
        batch_size=10,
        class_mode='binary')

autoencoder.summary()
autoencoder.fit(
        x=x_train,
        batch_size=10,
        epochs=50,
        validation_data=x_test,
        verbose=1)

print("\nFinished Training... \n")

autoencoder.save("256x256-model.h5")

print("\nFinished Saving... \n")

#show the before and after images
i = 0
decoded_imgs = autoencoder.predict(x_test[i][i])

print("Minimum:", np.min(decoded_imgs))
print("Maximum:", np.max(decoded_imgs))
n = 2

plt.figure(figsize=(4, 4))
img = x_test[i][i]
#original
ax = plt.subplot(2, n, i + 1)
plt.imshow(img[i].reshape(256, 256, 3))
ax.set_axis_off()

    #reconstruction
ax = plt.subplot(2, n, i + n + 1)
plt.imshow(decoded_imgs[i].reshape(256, 256), cmap='pyart_HomeyerRainbow')
ax.set_axis_off()
plt.show()
K.clear_session()
