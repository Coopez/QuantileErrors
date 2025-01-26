import torch

class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size[0]))
        for i in range(1,self.num_layers):
            self.layers.append(torch.nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError("Activation function not implemented")
    def forward(self, x):
        # flatten input
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = self.activation(layer(x))
        return x # maybe unsqueeze here