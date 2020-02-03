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
            
            ## Balancing the training data set
            # rounding the measurements
            rounded_measurements = []
            for measurement in measurements:
                rounded_measurements.append(round(measurement, 2))
            # Creating a dictionary where the rounded measurement is the key
            # and the value is the list of the images with that measurement
            data_dict = {key:None for key in set(rounded_measurements)}
            for i in range(0, len(rounded_measurements)):
                if data_dict[rounded_measurements[i]] == None:
                    data_dict[rounded_measurements[i]] = [i]
                else:
                    data_dict[rounded_measurements[i]].append(i)
                    
            # Setting the target number of samples for each key
            # as the maximum number of occurences for key
            target_samples = 0
            for value in data_dict.values():
                if len(value) > target_samples:
                    target_samples = len(value)

            # Reach the target number of samples for each key
            aug_images = []
            aug_measurements = []
            # generate new samples for each key until target value is reached
            for key in data_dict.keys():
                samples_to_create = target_samples - len(data_dict[key])
                for i in range(0, samples_to_create):
                    img = images[np.random.choice(data_dict[key])]
                    aug_images.append(img)
                    aug_measurements.append(key)

            # merge with existing data
            for i in range(0, len(images)):
                aug_images.append(images[i])
                aug_measurements.append(measurements[i])
            
            # Data augmentation flipping the images
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(aug_images, aug_measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(np.fliplr(image))
                augmented_measurements.append(-1.0 * measurement)
    
            X = np.array(augmented_images)
            y = np.array(augmented_measurements)
            
            yield sklearn.utils.shuffle(X, y)