import torch

class AmplitudeDataset(torch.utils.data.Dataset):
    def __init__(self, particles, amplitudes, sm, dtype):
        self.particles = torch.tensor(particles, dtype=dtype)
        self.amplitudes = torch.tensor(amplitudes, dtype=dtype)
        self.sm = torch.tensor(sm, dtype=dtype)

        self.len = len(self.particles)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.particles[idx], self.amplitudes[idx], self.sm[idx])
