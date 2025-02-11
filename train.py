import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from modules import EAG, IAG
from dataset import ActionSequenceDataset


class A2GenTrainer:
    def __init__(self, model, use_iag=False):
        self.model = model
        self.use_iag = use_iag
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, dataloader, num_epochs=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for action_seq, context_features, true_action_types, true_action_times in dataloader:
                self.model.train()
                self.optimizer.zero_grad()

                if self.use_iag:
                    predicted_action_types, predicted_action_times = self.model(action_seq, context_features)
                else:
                    predicted_action_types, predicted_action_times = self.model(action_seq, context_features)

                loss_cls = self.criterion_cls(predicted_action_types, true_action_types)
                loss_reg = self.criterion_reg(predicted_action_times, true_action_times)
                total_loss = loss_cls + loss_reg
                total_loss.backward()
                self.optimizer.step()

                total_loss += total_loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    sample_data = [
        (np.random.rand(10, 128), np.random.rand(128), np.random.randint(0, 10, 10), np.random.rand(10))
        for _ in range(1000)
    ]
    dataset = ActionSequenceDataset(sample_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = 128
    num_classes = 10
    model_eag = EAG(input_dim, num_classes)
    trainer = A2GenTrainer(model_eag, use_iag=False)
    trainer.train(dataloader, num_epochs=10)