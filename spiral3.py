import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import math
import matplotlib.pyplot as plt
import numpy as np


# Define functions to generate different shapes

@st.cache_data
def generate_spiral(n_samples=10000, noise=0.1, irregularity=0.05, circles=3):
    """
    Generates a 2D spiral dataset.

    Args:
        n_samples (int): The number of samples to generate (default: 10000).
        noise (float): The amount of noise to add to the spiral (default: 0.1).
        irregularity (float): The amount of irregularity to add to the spiral (default: 0.05).
        circles (int): The number of circles to generate (default: 3).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the 2D spiral dataset.
    """
    t = torch.linspace(0, 2 * math.pi * circles, n_samples)
    r = torch.linspace(0, 1, n_samples) + noise * torch.randn(n_samples)
    angle = t + irregularity * torch.randn(n_samples)
    x = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1)
    return x

@st.cache_data
def generate_spiral2(n_samples=1000, noise=0.1, irregularity=0.05, circles=5):
    t = torch.linspace(0, 2 * math.pi * circles, n_samples)
    r = torch.linspace(0, 1, n_samples) + noise**4 * torch.randn(n_samples)
    x = torch.stack([r * torch.cos(t+ irregularity* r), r * torch.sin(t + irregularity * r)], dim=1)
    return x

@st.cache_data
def generate_spiral3(n_samples=1000, noise=0.1, irregularity=0.05, circles=10):
    t = torch.linspace(0, 2 * math.pi * circles, n_samples)
    r = torch.linspace(0, 1, n_samples) + noise * torch.randn(n_samples)
    irregularities = irregularity * torch.randn(n_samples)
    x = torch.stack([r * torch.cos(t + irregularities), r * torch.sin(t + irregularities)], dim=1)
    return x

@st.cache_data
def generate_circle(n_samples=1000, noise=0):
    """
    Generates a 2D circle dataset.

    Args:
        n_samples (int): The number of samples to generate (default: 1000).
        noise (float): The amount of noise to add to the spiral (default: 0).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the 2D circle dataset.
    """
    t = torch.linspace(0, 2 * math.pi, n_samples)
    x = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    x += noise * torch.randn(n_samples, 2)
    return x

@st.cache_data
def generate_heart(n_samples=1000, noise=0):
    """
    Generates a 2D heart dataset.

    Args:
        n_samples (int): The number of samples to generate (default: 1000).
        noise (float): The amount of noise to add to the spiral (default: 0).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the 2D circle dataset.
    """
    t = torch.linspace(-math.pi, math.pi, n_samples)
    x = torch.stack([16 * torch.sin(t) ** 3, 13 * torch.cos(t) - 5 * torch.cos(2*t) - 2 * torch.cos(3*t) - torch.cos(4*t)], dim=1) / 20
    x += noise * torch.randn(n_samples, 2)
    return x



class ODE(nn.Module):
    """
    Implements a neural network that solves an ordinary differential equation (ODE).
    
    Args:
        dim (int): The number of dimensions in the input data.
        
    Attributes:
        func (ODEFunc): The ODE function that will be used to solve the differential equation.
        
    Methods:
        forward(x): Solves the ODE for the input data x.
        
    """
    def __init__(self, dim):
        """
        Initializes the ODE network.
        
        Args:
            dim (int): The number of dimensions in the input data.
        """
        super(ODE, self).__init__()
        self.func = ODEFunc(dim)

    def forward(self, x):
        """
        Solves the ODE for the input data x.
        
        Args:
            x (torch.Tensor): The input data.
        
        Returns:
            torch.Tensor: The solution to the ODE.
        """
        t = torch.linspace(0, 1, 2) #integration time
        out = odeint(self.func, x, t)
        return out[1]

class ODEFunc(nn.Module):
    """
    Implements the ODE function used to solve the differential equation.
    
    Args:
        dim (int): The number of dimensions in the input data.
        
    Attributes:
        net (nn.Sequential): The neural network used to solve the ODE.
        
    Methods:
        forward(t, x): Evaluates the ODE function for the given input data.
        
    """
    def __init__(self, dim):
        """
        Initializes the ODE function.
        
        Args:
            dim (int): The number of dimensions in the input data.
        """
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, t, x):
        """
        Evaluates the ODE function for the given input data.
        
        Args:
            t (torch.Tensor): The time values for which the ODE function should be evaluated.
            x (torch.Tensor): The input data.
        
        Returns:
            torch.Tensor: The output of the ODE function.
        """
        return self.net(x)



class ResBlock(nn.Module):
    """
    Defines a simple residual block.

    Args:
        dim (int): The input and output dimension of the block.

    Attributes:
        net (nn.Sequential): The feedforward neural network defining the residual block.

    Methods:
        forward(x): Computes the forward pass of the block.

    """
    def __init__(self, dim):

        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, x):
        """
        Computes the forward pass of the residual block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: The output tensor of the block of shape (batch_size, dim).

        """
        return self.net(x) + x


class ResNet(nn.Module):
    """
    Defines a simple feedforward neural network with residual connections.

    Args:
        dim (int): The input and output dimension of the network.

    Attributes:
        net (nn.Sequential): The feedforward neural network with residual connections.

    Methods:
        forward(x): Computes the forward pass of the network.

    """
    def __init__(self, dim):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )

    def forward(self, x):
        """
        Computes the forward pass of the residual neural network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: The output tensor of the network of shape (batch_size, dim).

        """
        return self.net(x)


# Define training loop
def train_model(model, train_data, val_data, optimizer, n_epochs=100):
    """
    Train a PyTorch model using the mean squared error (MSE) loss.

    Args:
    - model: a PyTorch model to train
    - train_data: a PyTorch tensor representing the training data
    - val_data: a PyTorch tensor representing the validation data
    - optimizer: a PyTorch optimizer used to update the model's parameters
    - n_epochs: an integer indicating the number of epochs to train the model for

    Returns:
    - train_loss_history: a list containing the training loss at each epoch
    - val_loss_history: a list containing the validation loss at each epoch
    """
    train_loss_history = []
    val_loss_history = []

    with st.empty() :
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(train_data), train_data)
            loss.backward()
            optimizer.step()
            train_loss = nn.MSELoss()(model(train_data), train_data).item()
            val_loss = nn.MSELoss()(model(val_data), val_data).item()
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            st.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


    return train_loss_history, val_loss_history

def main_spirale():
    """The main_spirale() function is a Streamlit app that generates shapes using Neural ODEs. 
    The app allows the user to select a shape type and customize the noise and irregularity levels. 
    The user can also choose to train a model on the generated shape using either a Neural ODE or ResNet model or compare the two.
    Finally, the app generates a new shape using the trained model.
    The function takes no arguments and returns nothing, as it is designed to run as a Streamlit app. 

- Generate a new shape: This section allows the user to select a shape type, noise level, and irregularity level. The shape type can be Spiral, Big Spiral, Circle, Heart, or Spiral with varying curves. Once the user has made their selection, the app generates a shape and displays it on a scatter plot.
- Train a model on the generated shape: This section allows the user to select a model type and learning rate. The user can choose to train either a Neural ODE or ResNet model or compare the two. The app then trains the selected model on the generated shape and displays the train and validation losses on a plot.
- Generate a new shape using the trained model: This section generates a new shape using the trained model. The user can choose the number of points in the new shape. The app then generates the new shape and displays it on a scatter plot along with the original shape for comparison.

Note that the app relies on several helper functions, including generate_spiral(), generate_circle(), generate_heart(), ODE(), ResNet(), train_model().
    """
    st.title("Shape Fitting with Neural ODEs")
    st.write("This app generates shapes then fits the shape by training a Resnet or Neural ODE. Choose the shape, with the noise and (the best visualization of spiral comes with noise=0 and irregularities=0.1). Then, choose whether you want Neural ODEs, Resnets or both. You'll also have to count five to ten minutes to train the Neural ODEs.")
    st.write("However, we included the code R.T.Q. Chen for a better generalisation of the fit (see sidebar).")
# Generate a new shape
    st.sidebar.subheader("Generate a new shape")
    shape_type = st.sidebar.selectbox("Select a shape type", ["Spiral","Big Spiral", "Circle", "Heart", "Spiral with varying curves (not very stable)"])
    noise_level = st.sidebar.slider("Noise level", 0.0, 0.1, 0.0, 0.01)
    irregularity_level = st.sidebar.slider("Irregularity level", 0.0, 1.0, 0.1, 0.1)
    if shape_type == "Spiral":
        shape = generate_spiral(noise=noise_level, irregularity=irregularity_level)
    elif shape_type == "Big Spiral":
        shape = generate_spiral2(noise=noise_level, irregularity=irregularity_level)
    elif shape_type == "Circle":
        shape = generate_circle(noise=noise_level)
    elif shape_type == "Heart":
        shape = generate_heart(noise=noise_level)
    else:
        shape= generate_spiral3(noise=noise_level, irregularity=irregularity_level)

    fig, ax = plt.subplots()

    ax.scatter(shape[:, 0], shape[:, 1], s=0.2)
    ax.set_aspect("equal")
    st.write("Generated shape")
    st.pyplot(fig)
# Train a model on the generated shape
    st.sidebar.subheader("Train a model on the generated shape")
    model_type = st.sidebar.selectbox("Select a model type", ["Neural ODE", "ResNet", "Compare the 2"])
    learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, 0.001)
    train_data = shape[:800]
    val_data = shape[800:]
    if model_type == "Neural ODE":
        model = ODE(dim=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loss, val_loss = train_model(model, train_data, val_data, optimizer, n_epochs=100)
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Validation Loss")
        ax.legend()
        st.pyplot(fig)
    elif model_type == "ResNet":
        model = ResNet(dim=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loss, val_loss = train_model(model, train_data, val_data, optimizer)
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Validation Loss")
        ax.legend()
        st.pyplot(fig)
    else: 
        model1 = ODE(dim=2)
        model2 = ResNet(dim=2)    
        optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
        st.write("Training Neural ODEs")
        train_loss1, val_loss1 = train_model(model1, train_data, val_data, optimizer1,n_epochs=100)
        placeholder = st.empty()
        st.write("Training ResNet")
        train_loss2, val_loss2 = train_model(model2, train_data, val_data, optimizer2)
        fig, ax = plt.subplots()
        ax.plot(train_loss1, label="Train Loss Neural ODE")
        ax.plot(val_loss1, label="Validation Loss Neural ODE")
        ax.plot(train_loss2, label="Train Loss ResNet")
        ax.plot(val_loss2, label="Validation Loss ResNet")
        ax.legend()
        st.pyplot(fig)
    


# Generate a new shape using the trained model
    st.sidebar.subheader("Generate a new shape using the trained model")
    num_points = 10000
    if model_type == "Neural ODE":
        xs = torch.zeros((num_points, 2))
        xs[0] = torch.tensor([0.1, 1.])

        with torch.no_grad() :
            for i in range(1, num_points) :
                x_out = model(xs[i-1])
                xs[i] = x_out
 


        fig, ax = plt.subplots()
        ax.scatter(xs[:, 0], xs[:, 1], s=0.2)
        ax.scatter(shape[:, 0], shape[:, 1], label="True", s=0.2)
        ax.set_aspect("equal")
        ax.legend()

        st.pyplot(fig)
    elif model_type == "ResNet":
        xs = torch.zeros((num_points, 2))
        xs[0] = torch.tensor([0.1, 0.])
        with torch.no_grad() :
            for i in range(1, num_points) :
                x_out = model(xs[i-1])
                xs[i] = x_out

        fig, ax = plt.subplots()
        ax.scatter(xs[:, 0], xs[:, 1], s=0.2)
        ax.scatter(shape[:, 0], shape[:, 1], label="True", s=0.2)
        ax.set_aspect("equal")
        ax.legend()

        st.pyplot(fig)

    else:   
        x1s = torch.zeros((num_points, 2))
        x2s = torch.zeros((num_points, 2))
        x1s[0] = torch.tensor([0.1, 0.])
        x2s[0] = torch.tensor([0.1, 0.])

        with torch.no_grad() :
            for i in range(1, num_points) :
                x1_out = model1(x1s[i-1])
                x1s[i] = x1_out                
                
                x2_out = model2(x2s[i-1])
                x2s[i] = x2_out
 
        fig, ax = plt.subplots()
        ax.scatter(x1s[:, 0], x1s[:, 1], label="Neural ODE", s=0.2)
        ax.scatter(x2s[:, 0], x2s[:, 1], label="ResNet", s=0.2)
        ax.scatter(shape[:, 0], shape[:, 1], label="True", s=0.2)
        ax.set_aspect("equal")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main_spirale()
       
