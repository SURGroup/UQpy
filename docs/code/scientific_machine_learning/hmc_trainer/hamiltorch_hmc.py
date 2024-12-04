"""Adapting the Hamiltorch notebook found at https://github.com/AdamCobb/hamiltorch/blob/master/notebooks/hamiltorch_Bayesian_NN_example.ipynb


"""

import torch
import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning as sml

hamiltorch.set_random_seed(123)
device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self, layer_sizes, loss="multi_class", bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=self.bias)
        self.l2 = sml.BayesianLinear(layer_sizes[1], layer_sizes[1])

    def forward(self, x):
        x = self.l1(x)

        return x


layer_sizes = [4, 3]
# net = Net(layer_sizes)
trunk_network = nn.Linear(1, 10)
branch_network = nn.Linear(2, 10)
net = sml.DeepOperatorNetwork(trunk_network, branch_network)

print(net)


from sklearn.datasets import load_iris
import numpy as np

np.random.seed(0)
data = load_iris()
x_ = data["data"]
y_ = data["target"]
N_tr = 10  # 50
N_val = 140
a = np.arange(x_.shape[0])
train_index = np.random.choice(a, size=N_tr, replace=False)
val_index = np.delete(a, train_index, axis=0)
x_train = x_[train_index]
y_train = y_[train_index]
x_val = x_[val_index][:]
y_val = y_[val_index][:]
x_m = x_train.mean(0)
x_s = x_train.std(0)
x_train = (x_train - x_m) / x_s
x_val = (x_val - x_m) / x_s
D_in = x_train.shape[1]
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_val = torch.FloatTensor(x_val)
y_val = torch.FloatTensor(y_val)
plt.scatter(x_train.numpy()[:, 0], y_train.numpy())

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

# Set hyperparameters for network
tau_list = []
tau = 1.0  # /100. # iris 1/10
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)


"""Sample using HMC"""
hamiltorch.set_random_seed(123)
net = Net(layer_sizes)
params_init = hamiltorch.util.flatten(net).to(device).clone()

step_size = 0.1
num_samples = 300
L = 20
tau_out = 1.0
params_hmc = hamiltorch.sample_model(
    net,
    x_train,
    y_train,
    params_init=params_init,
    num_samples=num_samples,
    step_size=step_size,
    num_steps_per_sample=L,
    tau_out=tau_out,
    tau_list=tau_list,
)
