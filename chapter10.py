import numpy as np

def normalize_inputs(data):
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    

    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std


data = np.array([[1, 2], [3, 4], [5, 6]])
normalized_data, mean, std = normalize_inputs(data)

print("Original Data:\n", data)
print("Normalized Data:\n", normalized_data)
print("Mean of each feature:\n", mean)
print("Standard Deviation of each feature:\n", std)