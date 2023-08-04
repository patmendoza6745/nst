import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
import time


# Define pre-trained model
pp = pprint.PrettyPrinter(indent=4)
img_size_w = 400
img_size_h = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size_w, img_size_h, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False

# Hyperparameters
alpha = 10
beta = 40
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
epochs = 20001
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Load in Content Image
content_image = np.array(Image.open("images/lil.jpeg").resize((img_size_h, img_size_w)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
 
# Load in Style Image
style_image =  np.array(Image.open("images/frida_style.jpg").resize((img_size_h, img_size_w)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

# Randomly initialize the Generated Image
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

# Load in pre-trained model
def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# Define content layer and build the model.
content_layer = [('block5_conv4', 1)]

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

# Save outputs of content and style images
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder

# Compute total cost

# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

# Assign the input of the model to be the "style" image 
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS)

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha = 10, beta = 40)
        
    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J

def train():
    for i in range(epochs):
        start = time.time()
        train_step(generated_image)
        elapsed = time.time() - start
        if i % 1 == 0:
            print(f"Epoch {i}, Time elapsed: {int(elapsed)} seconds.")
        if i % 2500 == 0 or i == epochs - 1:
            image = tensor_to_image(generated_image)
           # imshow(image)
            image.save(f"output/image_{i}.jpg")
            plt.show() 

def display_imgs():
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    imshow(content_image[0])
    ax.title.set_text('Content image')
    ax = fig.add_subplot(1, 3, 2)
    imshow(style_image[0])
    ax.title.set_text('Style image')
    ax = fig.add_subplot(1, 3, 3)
    imshow(generated_image[0])
    ax.title.set_text('Generated image')
    plt.show()

generated_image = tf.Variable(generated_image)
train()
display_imgs()