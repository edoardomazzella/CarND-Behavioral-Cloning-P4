import sklearn
import numpy as np
from sklearn.utils import shuffle
from scipy import ndimage

# Generators that:
#   - Splits the entire set of samples in batches of samples; 
#   - Considers center, left and right image of each sample of the batch
#     applying a correction factor on the steering angle;
#   - Augments the data by flipping each of the three image of a sample;
#   - Returns the augmented input and output batch data
def generator(samples, batch_size = 32, lr_correction_factor = 0.2):
    num_samples = len(samples)
    print('Number of samples passed to the generator: ', num_samples)
    # Loop forever so the generator never terminates
    while 1:
        # Shuffle the samples
        shuffle(samples)
        # Considering batches of samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            measurements = []
            # For each sample of the batch
            for batch_sample in batch_samples:
                # Iterate over the 3 images and store their steering angle
                for i in range(3):
                    name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    image = ndimage.imread(name)
                    images.append(image)
                    
                    # Applying correction factor to the left and right image
                    if i == 0:
                        measurement = float(batch_sample[3])
                    elif i == 1:
                        measurments = float(batch_sample[3]) + lr_correction_factor
                    else:
                        measurments = float(batch_sample[3]) - lr_correction_factor
                    measurements.append(measurement)
            
            # Data augmentation flipping the images
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(np.fliplr(image))
                augmented_measurements.append(-1.0 * measurement)
    
            X = np.array(augmented_images)
            y = np.array(augmented_measurements)
            
            yield sklearn.utils.shuffle(X, y)