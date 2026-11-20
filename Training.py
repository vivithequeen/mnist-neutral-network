import NeuralNetwork
import random
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

    def getAverageCost(self, weights_and_bias):
       

        train_labels = self.read_csv("mnist\\train_labels.csv")
        count = 0
        costSum = 0
        for n in train_labels[:20]:
            c = self.neuralNetwork.run_neural_network(weights_and_bias, f"train\\{n[0]}")
            
            cost = self.getCost(c, int(n[1]))
            count+=1
            costSum+=cost
        return costSum / count

    def write_weights_and_bias(self, new):
        dimensions = 28 * 28

        with open("weights_and_bias.csv", "w", encoding="utf-8") as f:
            # Write header row
            headers = [f"weight_{i}" for i in range(dimensions)] + ["bias"]
            f.write(",".join(headers) + "\n")

            # Write each row directly since 'new' already matches the CSV structure
            for row in new:
                f.write(",".join(str(val) for val in row) + "\n")




    def createCostFunction(self):
        init_weights_and_bias = self.read_csv("weights_and_bias.csv")

        init_average_cost = self.getAverageCost(init_weights_and_bias)
        print(f"Initial Average Cost: {init_average_cost}")

        best_weights = self.read_csv("weights_and_bias.csv")
        best_cost = self.getAverageCost(best_weights)
        print(f"Initial Average Cost: {best_cost}")

        steps = 100
        step_size = 0.05

        for i in range(steps):
            # Generate small positive AND negative random nudges
            candidate_weights = [
                [float(val) + random.uniform(-step_size, step_size) for val in row]
                for row in best_weights
            ]

            candidate_cost = self.getAverageCost(candidate_weights)

            if candidate_cost < best_cost:
                # Keep the improvement!
                best_weights = candidate_weights
                best_cost = candidate_cost
                print(f"Step {i}: Improved cost to {best_cost:.4f}")
            else:
                # Optional: slightly shrink step size if stuck
                step_size *= 0.995

        self.write_weights_and_bias(best_weights)
        print(f"Final Best Cost: {best_cost}")

        

            







