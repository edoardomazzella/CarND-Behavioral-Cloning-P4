from keras.layers import Lambda, Flatten, Dense, Conv2D, Cropping2D
from keras.models import Sequential

# Function that returns the nvidia NN model
def nvidiaModel():
    # Modeling NN architecture
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping = ((70, 25), (0, 0)), data_format = "channels_last"))
    model.add(Conv2D(24, (5, 5), strides = (2, 2), activation = "relu"))
    model.add(Conv2D(36, (5, 5), strides = (2, 2), activation = "relu"))
    model.add(Conv2D(48, (5, 5), strides = (1, 1), activation = "relu"))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation = "relu"))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation = "relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model