# Importing libraries
import csv
import sklearn
import matplotlib.pyplot as plt
from nvidia import nvidiaModel
from generator import generator
from math import ceil
from sklearn.model_selection import train_test_split
from keras.models import Model

# Importing lines of driving_log.csv file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # Appending samples but discarding the first line conatining the column names
    for cont, line in enumerate(reader):
        if cont != 0:
            samples.append(line)

# Splitting training and validation samples
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

# Generator parameters
batch_size = 8
lr_correction_factor = 0.2

# Creating training and validation generator
train_generator = generator(train_samples, batch_size = batch_size, 
                            lr_correction_factor = lr_correction_factor)
validation_generator = generator(validation_samples, batch_size = batch_size, 
                                 lr_correction_factor = lr_correction_factor)

# Creating the nvidia NN model
model = nvidiaModel()
# Compile the model with Adam optimizer and mse function
model.compile(optimizer = 'adam', loss = 'mse')
# Check the summary of this new model to confirm the architecture
model.summary()

# Number of epochs
epochs = 2

# Training the model with the generators
history_object = model.fit_generator(train_generator, 
                    steps_per_epoch = ceil(len(train_samples) / batch_size),
                    validation_data = validation_generator, 
                    validation_steps = ceil(len(validation_samples) / batch_size),
                    epochs = epochs, verbose = 1)

# Stores the plot of the training and validation loss for each epoch
plt.ioff()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.savefig("lossperepoch.png")

# Saving the model
model.save('model.h5')