import pytorch_lightning as pl
import torch

from .model import Dataset2Vec


class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        encoder: Dataset2Vec,
        gamma: float = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def training_step(self, batch, *args, **kwargs):
        metrics = self.__calculate_batch_metrics(batch, *args, **kwargs)
        self.log("train_loss", metrics["loss"], prog_bar=True, batch_size=len(batch))
        return metrics["loss"]

    def validation_step(self, batch, *args, **kwargs):
        metrics = self.__calculate_batch_metrics(batch, *args, **kwargs)
        self.log("val_loss", metrics["loss"], prog_bar=True, batch_size=len(batch))
        self.log("val_accuracy", metrics["accuracy"], prog_bar=True, batch_size=len(batch))
        return metrics["loss"]

    def __calculate_batch_metrics(self, batch, *args, **kwargs):
        similarities = []
        labels = []
        for obs in batch:
            X1, y1, X2, y2, label = obs
            emb1 = self.encoder(X1, y1)
            emb2 = self.encoder(X2, y2)
            similarities.append(torch.exp(-self.gamma * torch.norm(emb1 - emb2)))
            labels.append(label)
        labels = torch.Tensor(labels).to(torch.bool)
        similarities = torch.stack(similarities, 0)
        same_datasets = torch.where(labels)[0]
        different_datasets = torch.where(~labels)[0]
        loss = (
            -torch.log(similarities[same_datasets]).mean()
            - torch.log(1 - similarities[different_datasets]).mean()
        )
        accuracy = ((similarities >= 0.5) == labels).to(float).mean()
        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
