import numpy as np

def generate_mini_batches(X, y, batch_size):

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
 
    mini_batches = []
    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1, 0, 0, 0, 1])
batch_size = 2

mini_batches = generate_mini_batches(X, y, batch_size)

for idx, (batch_X, batch_y) in enumerate(mini_batches):
    print(f"Batch {idx + 1}  X:\n{batch_X}\nBatch {idx + 1}  y:\n{batch_y}")
