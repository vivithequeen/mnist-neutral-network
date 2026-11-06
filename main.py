import math
import os
import random
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

layers = 2
hidden_layer_size = 32
characters = 10
dimentions = 28*28

def sigmoid(x):
    return 1/(1+(math.e**-x))

def ReLU(x):
    return max(0, x)

def setup_weights_and_bias():
    with open("weights_and_bias.csv", "w", encoding="latin-1") as f:
        for i in range(dimentions):
            f.write(f"weight_{i},")
        f.write("bias")
        a = [dimentions, hidden_layer_size, hidden_layer_size, characters]
        paramater_count = 0
        for i in [1,2,3]:
            for j in range(a[i]):
                f.write("\n")
                for k in range(a[i-1]):
                    f.write(f"{((random.random() * 2) -1) * 1 },") # weight
                    paramater_count+=1
                f.write(f"{((random.random() * 2) -1) * 1 }") # bias
                paramater_count+=1
        print(paramater_count)
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
    return ReLU(weighted_sum + float(bias[0]))

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

def visualize_what_on_earth_is_happaning(image_array, values):
  

    image_2d = image_array.reshape((28, 28))


    weights_2d_1 = np.array(values, dtype=float).reshape((1, 10))

    fig, (ax3, ax1) = plt.subplots(1, 2, figsize=(10, 5))

    im3 = ax3.imshow(image_2d, cmap="gray")
    ax3.set_title("Input Image")
    ax3.axis("off")  

    im1 = ax1.imshow(weights_2d_1, cmap="viridis", aspect='auto')
    ax1.set_title("Weights Heatmap")
    ax1.set_xlabel("Weight Index")
    ax1.set_yticks([])

    fig.colorbar(im1, ax=ax1, orientation='vertical')

    plt.tight_layout()
    plt.show()

def visualize_guesses(x,y, percent_correct):

    # Generate random sample data


    # Create the scatter plot
    plt.figure(figsize=(6, 4))
    col = []
    for i in range(len(x)):
        if(x[i]==y[i]):
            col.append('royalblue')
        else:
            col.append('red')

    sizes = []
    for i in range(len(x)):
        sizes.append(x.count(y[i]) * 5 )
    plt.scatter(x, y, s = sizes, color=col, marker='o')

    # Add titles and labels
    plt.title(f'guess vs answer({percent_correct}% correct)')
    plt.xlabel('guess')
    plt.ylabel('answer')
    plt.grid(True)

    # Display the plot
    plt.show()

def run_neural_network(weights_and_bias, path):
    image_array = array_from_image(f"mnist\\test\\{path}")
    a = get_activation_value_layer(image_array,weights_and_bias, 0, hidden_layer_size)
    b = get_activation_value_layer(a,weights_and_bias, 1, hidden_layer_size)
    c = get_activation_value_layer(b, weights_and_bias, 2, hidden_layer_size)

    return c # gussed character


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

    temp_test_labels = test_labels[:1000]

    num_correct = 0

    all_eval = []
    all_correct = []

    for n in temp_test_labels:
        #print(n)
        c = run_neural_network(weights_and_bias, n[0])
        evaluated = c.index(max(c))
        #print(evaluated, n[1],"\n")
        all_eval.append(evaluated)
        all_correct.append(int(n[1]))
        if(evaluated == int(n[1])):
            num_correct+=1
    print(f"% correct:{(num_correct / len(temp_test_labels)) * 100}")
    return all_eval, all_correct, (num_correct / len(temp_test_labels)) * 100

#setup_weights_and_bias()

#x,y, percent_correct = evaluate_network()

#visualize_guesses(x,y, percent_correct)

#b = get_activation_value_layer(a, weights_and_bias, 1)
weights_and_bias = read_weights_and_bias()
num = 356
c = run_neural_network(weights_and_bias, f"{num}.png")
image_array = array_from_image(f"mnist\\test\\{num}.png")

print(c)
visualize_what_on_earth_is_happaning(image_array,c)