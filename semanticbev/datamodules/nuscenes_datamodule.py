from semanticbev.datamodules.components.nuscenes_data import compile_data

import pytorch_lightning as pl


class NuScenesDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Nuscenes dataset.

    This DataModule implements:
        - setup : things to do on every accelerator (GPU, TPU,...) in distributed mode
        - train_dataloader : the training dataloader
        - val_dataloader : the validation dataloader
        - test_dataloader : the test dataloader

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """


    def __init__(self, version, dataroot, data_aug_conf, grid_conf, batch_size, num_workers, pin_memory,
                 train_shuffle=True, prefetch_factor=4, update_intrinsics=True, parser_name='segmentationdata'):
        super().__init__()

        self.version = version
        self.dataroot = dataroot
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_shuffle = train_shuffle
        self.prefetch_factor = prefetch_factor
        self.update_intrinsics = update_intrinsics
        self.parser_name = parser_name

    def setup(self, stage):
        self.trainloader, self.valloader = compile_data(self.version,
                                                        self.dataroot,
                                                        data_aug_conf=self.data_aug_conf,
                                                        grid_conf=self.grid_conf,
                                                        bsz=self.batch_size,
                                                        nworkers=self.num_workers,
                                                        pin_memory=self.pin_memory,
                                                        parser_name=self.parser_name,
                                                        train_shuffle=self.train_shuffle,
                                                        prefetch_factor=self.prefetch_factor,
                                                        update_intrinsics=self.update_intrinsics
                                                        )

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader
