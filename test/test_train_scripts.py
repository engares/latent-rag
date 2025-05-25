import pytest
import torch
import types
from training import train_vae, train_dae, train_cae

class DummyLogger:
    def __init__(self):
        self.main = self
        self.train = self
    def info(self, *args, **kwargs):
        pass

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10, d=8):
        self.n = n; self.d = d
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return {"input": torch.randn(self.d), "target": torch.randn(self.d), "x": torch.randn(self.d), "y": torch.randn(self.d), "q": torch.randn(self.d), "p": torch.randn(self.d), "n": torch.randn(self.d)}

class DummyVAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        out = x + self.dummy_param
        return out, torch.zeros(x.size(0), 2), torch.zeros(x.size(0), 2)

class DummyDAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x + self.dummy_param

class DummyCAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def encode(self, x):
        return x + self.dummy_param
    def forward(self, x):
        return x + self.dummy_param

def patch_train_vae(monkeypatch):
    monkeypatch.setattr("training.train_vae.EmbeddingVAEDataset", lambda path: DummyDataset())
    monkeypatch.setattr("training.train_vae.VariationalAutoencoder", DummyVAE)
    monkeypatch.setattr("training.train_vae.torch.save", lambda *a, **k: None)
    monkeypatch.setattr("training.train_vae.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("utils.data_utils.split_dataset", lambda ds, val_split: (ds, ds))

def patch_train_dae(monkeypatch):
    monkeypatch.setattr("training.train_dae.EmbeddingDAEDataset", lambda path: DummyDataset())
    monkeypatch.setattr("training.train_dae.DenoisingAutoencoder", DummyDAE)
    monkeypatch.setattr("training.train_dae.torch.save", lambda *a, **k: None)
    monkeypatch.setattr("training.train_dae.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("utils.data_utils.split_dataset", lambda ds, val_split: (ds, ds))

def patch_train_cae(monkeypatch):
    monkeypatch.setattr("training.train_cae.EmbeddingTripletDataset", lambda path: DummyDataset())
    monkeypatch.setattr("training.train_cae.ContrastiveAutoencoder", DummyCAE)
    monkeypatch.setattr("training.train_cae.torch.save", lambda *a, **k: None)
    monkeypatch.setattr("training.train_cae.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("utils.data_utils.split_dataset", lambda ds, val_split: (ds, ds))

def test_train_vae_runs(monkeypatch):
    patch_train_vae(monkeypatch)
    train_vae.train_vae(
        dataset_path="dummy",
        input_dim=8,
        latent_dim=2,
        hidden_dim=4,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        model_save_path="/tmp/vae.pth",
        val_split=0.2,
        patience=1,
        device="cpu",
    )

def test_train_dae_runs(monkeypatch):
    patch_train_dae(monkeypatch)
    train_dae.train_dae(
        dataset_path="dummy",
        input_dim=8,
        latent_dim=2,
        hidden_dim=4,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        model_save_path="/tmp/dae.pth",
        val_split=0.2,
        patience=1,
        device="cpu",
        logger=DummyLogger(),
    )

def test_train_cae_runs(monkeypatch):
    patch_train_cae(monkeypatch)
    train_cae.train_cae(
        dataset_path="dummy",
        input_dim=8,
        latent_dim=2,
        hidden_dim=4,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        model_save_path="/tmp/cae.pth",
        val_split=0.2,
        patience=1,
        device="cpu",
        logger=DummyLogger(),
    )
