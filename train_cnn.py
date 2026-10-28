"""Tiny CNN forward/training smoke demo on synthetic tensors; NOT a trained production model."""
import torch
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(24, 4))
    def forward(self, x): return self.net(x)

if __name__ == '__main__':
    torch.manual_seed(42)
    model = TinyCNN()
    x = torch.rand(8,3,64,64)
    y = torch.randint(0,4,(8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward(); optimizer.step()
    print('One toy gradient update completed; loss:', round(loss.item(), 4))
