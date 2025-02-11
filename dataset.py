import torch
from torch.utils.data import DataLoader, Dataset


class ActionSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        action_seq, context_features, true_action_types, true_action_times = self.data[idx]
        return (
            torch.tensor(action_seq, dtype=torch.float32),
            torch.tensor(context_features, dtype=torch.float32),
            torch.tensor(true_action_types, dtype=torch.long),
            torch.tensor(true_action_times, dtype=torch.float32)
        )
