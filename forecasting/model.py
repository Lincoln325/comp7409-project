import torch
from torch import nn

# Build model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2 
output_dim = 1

# Here we define our model as a class
class V1(nn.Module):
    """
    A neural network model that uses a Long Short-Term Memory (LSTM) layer followed by a linear layer.
    
    Args:
        input_dim (int): The dimension of the input tensor (default: 1).
        hidden_dim (int): The dimension of the hidden state of the LSTM layer (default: 32).
        num_layers (int): The number of hidden layers in the LSTM layer (default: 2).
        output_dim (int): The dimension of the output tensor (default: 1).
    
    Example Usage:
        # Create an instance of the V1 class
        model = V1(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    
        # Generate some input data
        x = torch.randn(100, 10, 1)
    
        # Pass the input data through the model
        output = model(x)
    
        # Print the output
        print(output)
    """
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(V1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Performs the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out