import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .efi_dataset import EfiDataset
from dataset.ocnn_utils import CustomCollateBatch

class EfiDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the EFI point cloud dataset.
    Handles data preparation, setup, and creation of DataLoaders.
    """
    def __init__(self, cfg, args):
        super().__init__()
        self.cfg = cfg # Full dataset config (e.g., root, csv_path)
        self.args = args # Command line arguments (e.g., batch_size, num_point)
        self.num_workers = 4

        # Use custom collate function for OCNN model
        if args.model == 'ocnn':
            self.collate_fn = CustomCollateBatch(batch_size=args.batch_size, target_len=1)
        else:
            self.collate_fn = None

    def prepare_data(self):
        """
        If 'process_data' is true, this triggers the pre-processing step: prepare_data(self).
        """
        if getattr(self.args, "process_data", False):
            for split in ["train", "val", "test"]:
                print(f"--- Preprocessing {split} ---")
                EfiDataset(
                    root=self.cfg['root'],
                    csv_path=self.cfg['csv'],
                    label_col=self.cfg['label_col'],
                    task_type=self.cfg['task'],
                    split=split,
                    num_points=self.args.num_point,
                    classes_list=self.cfg.get('classes', None),
                    process_data=True,
                    use_normals=self.cfg['use_normal'],
                    use_fps=self.cfg['use_fps'],
                )

    def setup(self, stage=None):
        common_params = dict(
            root=self.cfg['root'], 
            csv_path=self.cfg['csv'], 
            label_col=self.cfg['label_col'], 
            task_type=self.cfg['task'],
            num_points=self.args.num_point,
            classes_list=self.cfg['classes'],
            process_data=False, # rely on cache or on-the-fly loading
            use_normals=self.cfg['use_normal'],
            use_fps=self.cfg['use_fps']
        )

        if stage in (None, "fit"):
            self.train_dataset = EfiDataset(split="train", model_name=self.args.model, **common_params)
            self.val_dataset = EfiDataset(split="val", model_name=self.args.model, **common_params)
            
        if stage in (None, "test", "predict"):
            self.test_dataset = EfiDataset(split="test", model_name=self.args.model, **common_params)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )