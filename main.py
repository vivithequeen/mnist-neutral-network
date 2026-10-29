import math
import os
import random
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

layers = 1
hidden_layer_size = 16
characters = 10
dimentions = 28*28

def sigmoid(x):
    return 1/(1+(math.e**-x))



def setup_weights_and_bias():
    with open("weights_and_bias.csv", "w", encoding="latin-1") as f:
        for i in range(dimentions):
            f.write(f"weight_{i},")
        f.write("bias")
        a = [dimentions, hidden_layer_size, hidden_layer_size, characters]
        for i in [1,2,3]:
            for j in range(a[i]):
                f.write("\n")
                for k in range(a[i-1]):
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
    for n in range(len(prev)):
        weighted_sum+=float(prev[n]) * float(weights[n])
    return sigmoid(weighted_sum + float(bias[0]))

def get_activation_value_layer(prev, weights_and_bias, layer, layer_size):
    layer_activation_value = []
    for n in weights_and_bias[(layer_size)*layer:(layer_size)*(layer+1)]:
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

def run_neural_network(weights_and_bias, path):
    image_array = array_from_image(f"mnist\\test\\{path}")
    a = get_activation_value_layer(image_array,weights_and_bias, 0, hidden_layer_size)
    b = get_activation_value_layer(a,weights_and_bias, 1, hidden_layer_size)
    c = get_activation_value_layer(b, weights_and_bias, 2, characters)
    print(c)
    print(c.index(max(c)))
    return c.index(max(c)) # gussed character


def read_test_labels():
    
    with open('mnist/test_labels.csv',mode='r', encoding='utf-8') as file:
        master_array = []
        lines = file.readlines()
        for line in lines:
            l = line.strip().split(",")
            
            master_array.append(l)
    master_array.pop(0)
    return master_array

def evaluate_network():
    test_labels = read_test_labels()
    weights_and_bias = read_weights_and_bias()

    temp_test_labels = test_labels[:100]

    num_correct = 0
    for n in temp_test_labels:
        print(n)
        evaluated = run_neural_network(weights_and_bias, n[0])
        #print(evaluated, n[1],"\n")
        if(evaluated == int(n[1])):
            num_correct+=1
    print(f"% correct:{(num_correct / len(temp_test_labels)) * 100}")

setup_weights_and_bias()

evaluate_network()


#b = get_activation_value_layer(a, weights_and_bias, 1)

#visualize_what_on_earth_is_happaning(image_array,a,b)