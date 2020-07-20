from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pyart
import glob


input_img = Input(shape=(200, 200, 3))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
decoded = Flatten()(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#autoencoder = load_model("autoencoder_model.h5")

#train the model
train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        featurewise_std_normalization=True,
        featurewise_center=True)

x_train = train_datagen.flow_from_directory(
        '../data/images/',
        target_size=(200, 200),
        batch_size=10,
        class_mode='binary')
x_test = train_datagen.flow_from_directory(
        '../data/images/',
        target_size=(200, 200),
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

# saving whole model
autoencoder.save('autoencoder_model.h5')

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
plt.imshow(img[i].reshape(200, 200, 3))
ax.set_axis_off()

    #reconstruction
ax = plt.subplot(2, n, i + n + 1)
plt.imshow(decoded_imgs[i].reshape(200, 200), cmap='pyart_HomeyerRainbow')
ax.set_axis_off()
plt.show()
K.clear_session()