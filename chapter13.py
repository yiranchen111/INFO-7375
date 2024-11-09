import numpy as np

# Original image (6 x 6)
image = np.array([[1, 2, 3, 0, 1, 2],
                  [4, 5, 6, 1, 0, 3],
                  [7, 8, 9, 2, 1, 0],
                  [1, 3, 2, 4, 5, 6],
                  [0, 1, 3, 7, 8, 9],
                  [2, 4, 6, 5, 3, 1]])

# Filter (3 x 3)
filter = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])


output_dim = 4
output = np.zeros((output_dim, output_dim))


for i in range(output_dim):
    for j in range(output_dim):
        region = image[i:i+3, j:j+3]  
        output[i, j] = np.sum(region * filter)  

print("Convoluted Image:")
print(output)