
import numpy as np
import matplotlib.pyplot as plt
import torch


dataset_x = torch.arange(0, 100, dtype=torch.float32)
dataset_x = dataset_x.reshape(-1, 1)
dataset_y = 2.6 * dataset_x + 4 + torch.randn_like(dataset_x)*20
dataset_y = dataset_y.reshape(-1, 1)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.register_parameter('w', torch.nn.Parameter(torch.randn(1)))
        self.register_parameter('b', torch.nn.Parameter(torch.randn(1)))

    def forward(self, x):
        return self.w * x + self.b


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_func = torch.nn.MSELoss()

for epoch in range(10000):

    for batch in dataloader:

        y_pred = model(dataset_x)
        loss = loss_func(y_pred, dataset_y)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


print(model.w, model.b)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(dataset_x, dataset_y, 'ro')
ax.plot(dataset_x, model(dataset_x).detach().numpy(), 'g-')
plt.show()
