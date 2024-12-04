import torch
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml
from train_fno import FourierNeuralOperator, BurgersDataset

model = FourierNeuralOperator()
model.load_state_dict(torch.load("fno_state_dict.pt", weights_only=True))

print(sml.FeedForwardNeuralNetwork(model).count_parameters())

train_dataset = BurgersDataset(
    "initial_conditions_train.pt",
    "burgers_solutions_train.pt",
)
test_dataset = BurgersDataset(
    "initial_conditions_test.pt",
    "burgers_solutions_test.pt",
)


x = torch.linspace(0.0, 1.0, 256)  # spacial domain
fig, ax = plt.subplots()
initial_condition, burgers_solution = train_dataset[10:11]
with torch.no_grad():
    prediction = model(initial_condition)

initial_condition = initial_condition.squeeze()
burgers_solution = burgers_solution.squeeze()
prediction = prediction.squeeze()
ax.plot(x, initial_condition, label="$y(x,0)$", color="black", linestyle="dashed")
ax.plot(
    x,
    burgers_solution,
    label="$y(x, 0.5)$",
    color="black",
)
ax.plot(x, prediction, label="FNO $\hat{y}(x, 0.5)$", color="tab:blue")
ax.set_title("Deterministic FNO on Training Data", loc="left")
ax.set(ylabel="$y(x, t)$", xlabel="Time $t$")
ax.legend(fontsize="small", ncols=2)

plt.show()
