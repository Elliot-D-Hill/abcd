import polars as pl
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def make_tensor_dataset(dataset: pl.DataFrame):
    data = []
    for subject in dataset.partition_by(
        "src_subject_id", maintain_order=True, include_key=False
    ):
        label = subject.drop_in_place("y_{t+1}").to_torch().float()
        features = subject.to_torch(dtype=pl.Float32)
        data.append((features, label))
    return data


class RNNDataset(Dataset):
    def __init__(self, dataset: pl.DataFrame) -> None:
        self.dataset = make_tensor_dataset(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, labels = self.dataset[index]
        return features, labels


def collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=float("nan"))
    return padded_features, padded_labels


class ABCDDataModule(LightningDataModule):
    def __init__(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame,
        test: pl.DataFrame,
        batch_size: int,
        dataset_class,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dataset = dataset_class(dataset=train)
        self.val_dataset = dataset_class(dataset=val)
        self.test_dataset = dataset_class(dataset=test)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_names = train.drop(["src_subject_id", "y_{t+1}"]).columns

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            collate_fn=collate_fn,
        )
