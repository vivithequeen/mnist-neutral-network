import math
import os
import random
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

layers = 1
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
        #print(weighted_sum)
        weighted_sum+=float(prev[n]) * float(weights[n])
    print(weighted_sum)
    return sigmoid(weighted_sum)

def read_weights_and_bias():
    with open('weights_and_bias.csv', mode='r', encoding='utf-8') as file:
        master_array = []
        lines = file.readlines()
        for line in lines:
            l = line.strip().split(",")
            
            master_array+=l
    return master_array[dimentions+1:]

def visualize_what_on_earth_is_happaning(image_array, weights_and_bias):
    image_2d = image_array.reshape((28,28))
    weights_2d = np.array(weights_and_bias[:dimentions], dtype=float).reshape((28,28))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(weights_2d, cmap="bwr")
    ax1.set_title("Weights")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(image_2d, cmap="gray")
    ax2.set_title("Input Image")
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

setup_weights_and_bias()     

weights_and_bias = read_weights_and_bias()


image_array = array_from_image("mnist\\train\\0.png")

print(get_activation_value(image_array, weights_and_bias[:dimentions], weights_and_bias[dimentions:dimentions+1]))


visualize_what_on_earth_is_happaning(image_array, weights_and_bias)