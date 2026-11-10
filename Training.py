import NeuralNetwork

class Training:
    neuralNetwork = NeuralNetwork.NeuralNetwork()
        




    def read_csv(self,file):

        with open(file,mode='r', encoding='utf-8') as file:
            master_array = []
            lines = file.readlines()
            for line in lines:
                l = line.strip().split(",")

                master_array.append(l)
        master_array.pop(0)
        return master_array
    

        
    def getCost(self, network_output, intended_number):

        num_array =  [0] * len(network_output)
        

        num_array[intended_number] = 1

        cost = 0
        for i in range(len(network_output)):
            cost+=(network_output[i] - num_array[i])**2

        return cost

    def getAverageCost(self):
        weights_and_bias = self.read_csv("weights_and_bias.csv")

        train_labels = self.read_csv("mnist\\train_labels.csv")
        count = 0
        costSum = 0
        for n in train_labels[:9000]:
            c = self.neuralNetwork.run_neural_network(weights_and_bias, f"train\\{n[0]}")
            
            cost = self.getCost(c, int(n[1]))
            count+=1
            costSum+=cost
        return costSum / count





