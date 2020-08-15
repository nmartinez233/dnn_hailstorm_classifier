from keras.models import Model, load_model

autoencoder = load_model('autoencoder_weights.h5')
autoencoder.summary()

new_model = Model(autoencoder.inputs, autoencoder.layers[-16].output)
new_model.summary()

new_model.save('encoder.h5')