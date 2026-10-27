import math
import os
import random
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

layers = 2
characters = 10
dimentions = 28*28

def sigmoid(x):
    return 1/(1+(math.e**-x))



def setup_weights_and_bias():
    with open("weights_and_bias.csv", "w", encoding="latin-1") as f:
        for i in range(dimentions):
            f.write(f"weight_{i},")
        f.write("bias")
        
        for i in range(dimentions * layers):
            f.write("\n")
            for j in range(dimentions):
                f.write(f"{((random.random() * 2) -1) * 1 },") # weight
            f.write(f"{((random.random() * 2) -1) * 1 }") # bias

def array_from_image(filename):
    img = Image.open(filename)
    img.convert('L')

    img_array = np.asarray(img)
    pixel_1d = img_array.flatten()

    return pixel_1d / 255

def get_activation_value(prev, weights, bias):
    weighted_sum = 0
    for n in range(dimentions):
        weighted_sum+=float(prev[n]) * float(weights[n])
    return sigmoid(weighted_sum + float(bias[0]))

def get_activation_value_layer(prev, weights_and_bias, layer):
    layer_activation_value = []
    print(prev)
    for n in weights_and_bias[dimentions*layer:dimentions*(layer+1)]:
        layer_activation_value.append(get_activation_value(prev, n[:dimentions], n[:-1]))
    return layer_activation_value

def read_weights_and_bias():
    with open('weights_and_bias.csv', mode='r', encoding='utf-8') as file:
        master_array = []
        lines = file.readlines()
        for line in lines:
            l = line.strip().split(",")
            
            master_array.append(l)
    master_array.pop(0)
    
    return master_array

def visualize_what_on_earth_is_happaning(image_array, weights_and_bias_layer_1,weights_and_bias_layer_2 ):
    image_2d = image_array.reshape((28,28))
    weights_2d_1 = np.array(weights_and_bias_layer_1[:dimentions], dtype=float).reshape((28,28))
    weights_2d_2 = np.array(weights_and_bias_layer_2[:dimentions], dtype=float).reshape((28,28))

    fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))

    im3 = ax3.imshow(image_2d, cmap="gray")
    ax3.set_title("Input Image")
    fig.colorbar(im3, ax=ax3)

    im1 = ax1.imshow(weights_2d_1, cmap="bwr")
    ax1.set_title("Weights, layer 1")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(weights_2d_2, cmap="bwr")
    ax2.set_title("Weights, layer 2")
    fig.colorbar(im2, ax=ax2)


    plt.tight_layout()
    plt.show()

setup_weights_and_bias()     

weights_and_bias = read_weights_and_bias()


image_array = array_from_image("mnist\\train\\0.png")
a = get_activation_value_layer(image_array,weights_and_bias, 0)
b = get_activation_value_layer(a, weights_and_bias, 1)

visualize_what_on_earth_is_happaning(image_array,a,b)