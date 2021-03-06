import numpy as np

class NN(object):

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        
        np.random.seed(1)
        
        pass

    
     # query the neural network
    def forward(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors) 
        

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))



def main():
    nn = NN(2,2,2, 0.15)

    epochs = 10000
    trainData = [
        ([0,0],[1,1]),
        ([1,0],[0,1]),
        ([0,1],[1,0]),
        ([1,1],[1,0])
    ]

    for e in range(epochs):
        for inputs, targetData in trainData:
            nn.train(inputs, targetData)

    print(nn.forward([0,0]))

if __name__ == "__main__":
    main()
