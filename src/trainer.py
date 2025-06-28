import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from src.models.faster_rcnn_mobilenet import FasterRCNNMobileNet


# ---- Dummy Detection Dataset ----
class DummyDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, 400, 400)  # Dummy 3-channel image
        target = {
            'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),  # Dummy box
            'labels': torch.tensor([1], dtype=torch.int64)  # Dummy class label
        }
        return image, target

    def __len__(self):
        return self.num_samples


# ---- Lightning Module Wrapper ----
class LightningFasterRCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = FasterRCNNMobileNet(input_channels=3, num_classes=num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = zip(*batch)
        images = list(img.to(self.device) for img in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer


# ---- Data Module (Optional but Recommended) ----
class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = DummyDetectionDataset()

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))  # Required for detection models
        )


# ---- Training Run ----
if __name__ == "__main__":
    num_classes = 2  # Example: 1 object class + background
    model = LightningFasterRCNN(num_classes=num_classes)
    data_module = DetectionDataModule(batch_size=2)

    trainer = Trainer(max_epochs=5, accelerator="auto")
    trainer.fit(model, datamodule=data_module)
