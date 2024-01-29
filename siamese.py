# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:56:49 2024

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.optim as optim

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Shared Convolutional Layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(11264, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward_one(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

def contrastive_loss(output1, output2, label, margin=2.0):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

# Sample Data
data_room1 = torch.randn((100, 1, 50))  # Assuming 100 samples from room 1
data_room2 = torch.randn((100, 1, 50))  # Assuming 100 samples from room 2
labels = torch.randint(0, 2, (100,))  # Binary labels indicating whether the actions are the same

# Model, Loss, and Optimizer
siamese_net = SiameseNetwork()
criterion = contrastive_loss
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output1, output2 = siamese_net(data_room1, data_room2)
    loss = criterion(output1, output2, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
