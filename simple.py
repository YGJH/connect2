import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mlp, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def main():
    model = mlp(4, 8 , 2)
    print(model)

    input_data = torch.tensor([[0.0, 1.0, 2.0 , 3.0], [5.0, 6.0, 3.0 ,2.0]])
    target = torch.tensor([sum(input_data[0]),  sum(input_data[1])])
    print(target)



    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    from tqdm.auto import tqdm
    with tqdm(total=10000) as phar:    
        for epoch in range(10000):
            if epoch % 10 == 0:
                phar.update(10)
            model.train()
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

    print(f"Loss: {loss.item()}")
    print(model(input_data))


if __name__ == "__main__":
    main()
