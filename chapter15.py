import tensorflow as tf
import numpy as np

def apply_convolution_tf(image, kernel, mode="depthwise"):
   
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    kernel_tensor = tf.convert_to_tensor(kernel, dtype=tf.float32)

    if mode == "depthwise":
        output = tf.nn.depthwise_conv2d(image_tensor, kernel_tensor, strides=[1, 1, 1, 1], padding="VALID")
    elif mode == "pointwise":
        output = tf.nn.conv2d(image_tensor, kernel_tensor, strides=[1, 1, 1, 1], padding="VALID")
    else:
        raise ValueError("Mode should be either 'depthwise' or 'pointwise'")
    
    return output

image = np.random.rand(1, 4, 4, 2).astype(np.float32)


depthwise_kernel = np.random.rand(3, 3, 2, 1).astype(np.float32)


pointwise_kernel = np.random.rand(1, 1, 2, 3).astype(np.float32)

# Perform depthwise convolution
depthwise_result = apply_convolution_tf(image, depthwise_kernel, mode="depthwise")
print("Depthwise Convolution Result:\n", depthwise_result.numpy())

# Perform pointwise convolution
pointwise_result = apply_convolution_tf(image, pointwise_kernel, mode="pointwise")
print("Pointwise Convolution Result:\n", pointwise_result.numpy())