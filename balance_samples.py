import numpy as np

# Function for balancing samples
def balance_samples(samples):
    # rounding steering measurements and collecting them in a list
    rounded_measurements = [] 
    for sample in samples:
        rounded_measurements.append(round(float(sample[3]), 4))

    # Creating a dictionary where the rounded measurement is the key
    # and the value is a list of the sample indexes with that measurement
    data_dict = {key : None for key in set(rounded_measurements)}
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
    augmented_samples =[]
    # generate new samples for each key until target value is reached
    for key in data_dict.keys():
        samples_to_create = target_samples - len(data_dict[key])
        for i in range(0, samples_to_create):
            line = samples[np.random.choice(data_dict[key])]
            augmented_samples.append(line)

    # merge with existing data
    for sample in samples:
        augmented_samples.append(sample)
    
    return augmented_samples