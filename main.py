# VolRecon

import argparse
from re import I
from stat import UF_OPAQUE
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.model_summary import ModelSummary


from code.model import VolRecon
from code.dataset.dtu_train import MVSDataset
from code.dataset.dtu_test_sparse import DtuFitSparse
from code.dataset.general_fit import GeneralFit

PI = math.pi
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------- main function
if __name__ == "__main__":

    seed_everything(0, workers=True)    

    # -------------------------------- args for training and models ---------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', dest='root_dir', type=str,
        help='directory of training dataset')
    parser.add_argument('--load_ckpt', dest='load_ckpt', type=str, default=False,
        help='load pretrained lightning ckpt')
    parser.add_argument('--train_ray_num', dest='train_ray_num', type=int, default=1024,
        help='ray number in one image')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
        help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2,
        help='batch size')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=16,
        help='max num of epochs')
    parser.add_argument('--val_only', dest='val_only', action="store_true",
        help='only validate')

    parser.add_argument('--volume_reso', dest='volume_reso', type=int, default=96, 
        help="3D feature volume resolution") # set as 0 to disable

    parser.add_argument('--coarse_sample', dest='coarse_sample', type=int, default=64,
        help='number of coarse samples during training')
    parser.add_argument('--fine_sample', dest='fine_sample', type=int, default=64,
        help='number of fine samples during training')
    # loss weights
    parser.add_argument('--weight_rgb', dest='weight_rgb', type=float, default=1.0)
    parser.add_argument('--weight_depth', dest='weight_depth', type=float, default=1.0)
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save checkpoints/logs')

    # -------------------------------- args for testing --------------------------------
    parser.add_argument('--test_dir', dest='test_dir', type=str,
        help='directory of test dataset')
    parser.add_argument('--out_dir', dest='out_dir', type=str,
        help='directory of to save test result')
    parser.add_argument('--extract_geometry', dest='extract_geometry', action='store_true', 
        help='if you only want to extract geometry')
    parser.add_argument('--test_general', dest='test_general', action='store_true', 
        help='test on custom dataset')
    
    parser.add_argument('--test_ray_num', dest='test_ray_num', type=int, default=1200)
    parser.add_argument('--test_sample_coarse', dest='test_sample_coarse', type=int, default=64)
    parser.add_argument('--test_sample_fine', dest='test_sample_fine', type=int, default=64)
    parser.add_argument('--test_coarse_only', dest='test_coarse_only', action="store_true",
        help='only use coarse samples during testing')
    parser.add_argument('--test_n_view', dest='test_n_view', type=int, default=3)

    parser.add_argument('--set', dest='set', type=int, default=0,
        help='two sets are provided by SparseNeuS')

    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_workers = 12
    devices = [0]

    # -------------------------------- dataset ----------------------------------------
    if not args.extract_geometry:
        # training
        dtu_dataset_train = MVSDataset(            
                root_dir=args.root_dir,
                split="train",
                split_filepath="code/dataset/dtu/lists/train.txt",
                pair_filepath="code/dataset/dtu/dtu_pairs.txt",
                n_views=5,
                )

        dtu_dataset_val = MVSDataset(            
                root_dir=args.root_dir,
                split="test",
                split_filepath="code/dataset/dtu/lists/test.txt",
                pair_filepath="code/dataset/dtu/dtu_pairs.txt",
                n_views=5,
                test_ref_views = [23],  # only use view 23
                )

        print("dtu_dataset_train:", len(dtu_dataset_train))
        print("dtu_dataset_val:", len(dtu_dataset_val))

        dataloader_train = DataLoader(dtu_dataset_train,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=True)  
        dataloader_val = DataLoader(dtu_dataset_val,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=False)  
    else:
        dataloader_test = []
        # dtu, 15 test scenes
        if not args.test_general:
            for scan in [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]:
                
                dataset_tmp = DtuFitSparse(root_dir=args.test_dir, 
                                    split="test", 
                                    scan_id='scan%d'%scan, 
                                    n_views=args.test_n_view,
                                    set=args.set)
                dataloader_tmp = DataLoader(dataset_tmp,
                                                batch_size=1, 
                                                num_workers=1, 
                                                shuffle=False)  
                dataloader_test.append(dataloader_tmp)
        else:
            for scan in ["general"]:
                
                dataset_tmp = GeneralFit(root_dir=args.test_dir, 
                                    scan_id=scan, 
                                    n_views=args.test_n_view)
                dataloader_tmp = DataLoader(dataset_tmp,
                                                batch_size=1, 
                                                num_workers=1, 
                                                shuffle=False)  
                dataloader_test.append(dataloader_tmp)

    # -------------------------------- lightning module -------------------------------
    if args.load_ckpt:
        volrecon = VolRecon.load_from_checkpoint(checkpoint_path=args.load_ckpt, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        volrecon = VolRecon(args)
    

    logger = WandbLogger(
        name = "volrecon"+args.logdir.rsplit('/')[-1],
        save_dir = args.logdir,
        offline=True,
    )

    # -------------------------------- trainer ---------------------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if device=="cuda" else "cpu", 
        devices=devices,
        strategy = "ddp",
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1, 
        logger=logger,
        num_sanity_val_steps=1,
        )

    
    ModelSummary(volrecon, max_depth=1)

    # -------------------------------- train or/and testing --------------------------------
    if not args.extract_geometry:
        if args.val_only:
            print("[only validation]")
            trainer.validate(volrecon, dataloader_train)
        else:
            print("[start training]")
            trainer.fit(volrecon, dataloader_train, dataloader_val)
    else:
        for dataloader_test1 in tqdm(dataloader_test):
            trainer.validate(volrecon, dataloader_test1)

    print("end")