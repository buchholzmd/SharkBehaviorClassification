from torch.utils.data import Dataset

class SharkBehaviorDataset(Dataset):
    def __init__(self, data, labels=None, train=True):
        self.data = data
        self.train = train
        if self.train:
            self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = None
        if self.train:
            sample_label = self.labels[idx]
            return (sample, sample_label)
        return sample
    