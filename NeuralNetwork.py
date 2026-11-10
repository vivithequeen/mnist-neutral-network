import math

from PIL import Image
import numpy as np

class NeuralNetwork:

    layers = 2
    hidden_layer_size = 32
    characters = 10
    dimentions = 28*28


    def sigmoid(self, x):
        return 1/(1+(math.e**-x))

    def ReLU(self, x):
        return max(0, x)


    def __init__(self):
        pass
    
        
    def array_from_image(self,filename):
        img = Image.open(filename)
        img.convert('L')
    
        img_array = np.asarray(img)
        pixel_1d = img_array.flatten()
    
        return pixel_1d / 255


    
    def get_activation_value(self, prev, weights, bias):
        weighted_sum = 0
        for n in range(len(prev)):
            weighted_sum+=float(prev[n]) * float(weights[n])
        return self.ReLU(weighted_sum + float(bias[0]))
    
    def get_activation_value_layer(self, prev, weights_and_bias, layer, layer_size):
        layer_activation_value = []
        for n in weights_and_bias[(layer_size) * layer:(layer_size)*(layer+1)]:
            layer_activation_value.append(self.get_activation_value(prev, n[:self.dimentions], n[:-1]))
        return layer_activation_value
    

        
    def run_neural_network(self,weights_and_bias, path):
        image_array = self.array_from_image(f"mnist\\{path}")
        a = self.get_activation_value_layer(image_array,weights_and_bias, 0, self.hidden_layer_size)
        b = self.get_activation_value_layer(a,weights_and_bias, 1, self.hidden_layer_size)
        c = self.get_activation_value_layer(b, weights_and_bias, 2,self.hidden_layer_size)

        return c # gussed character