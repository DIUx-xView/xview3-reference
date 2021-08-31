import configparser
import os
from random import Random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data

from constants import (
    NODATA_BATHYMETRY,
    NODATA_OWIMASK,
    NODATA_OWIWINDIDRECTION,
    NODATA_OWIWINDSPEED,
    NODATA_VH_DB,
    NODATA_VV_DB,
)
from dataloader import XView3Dataset
from engine import evaluate, train_one_epoch
from utils import collate_fn, xView3BaselineModel

nodata_dict = {
    "vv": NODATA_VV_DB,
    "vh": NODATA_VH_DB,
    "bathymetry": NODATA_BATHYMETRY,
    "wind_direction": NODATA_OWIWINDIDRECTION,
    "wind_speed": NODATA_OWIWINDSPEED,
    "mask": NODATA_OWIMASK,
}


def create_datasets(
    train_data_root,
    train_detect_file,
    train_chips_path,
    val_data_root,
    val_detect_file,
    val_chips_path,
    overwrite_preproc=False,
    num_workers=1,
    channels=["vh", "vv", "bathymetry", "wind_direction"],
    is_distributed=False,
):
    # Getting list of images with detects
    dfs, split_ids, roots = {}, {}, {}
    dfs["train"] = pd.read_csv(train_detect_file).dropna(subset=["is_vessel"])
    dfs["val"] = pd.read_csv(val_detect_file).dropna(subset=["is_vessel"])

    roots["train"] = train_data_root
    roots["val"] = val_data_root

    for split in dfs:
        # Currently handles ESA ID or scene_id
        split_ids[split] = [
            a.strip("\n").strip("/").split("/")[-1][:67]
            for a in os.listdir(roots[split])
        ]

        # Populate with any scenes that cause issues during training
        problem_scenes = []
        for problem_scene in problem_scenes:
            if problem_scene in split_ids:
                split_ids.remove(problem_scene)

    print(
        f"{len(split_ids['train'])} train IDs and {len(split_ids['val'])} validation IDs"
    )

    # TODO: Add input checking on channels

    train_data = XView3Dataset(
        train_data_root,
        None,
        "train",
        chips_path=train_chips_path,
        detect_file=train_detect_file,
        scene_list=split_ids["train"],
        background_frac=0.5,
        overwrite_preproc=overwrite_preproc,
        channels=channels,
        num_workers=num_workers,
    )

    val_data = XView3Dataset(
        val_data_root,
        None,
        "val",
        chips_path=val_chips_path,
        detect_file=val_detect_file,
        scene_list=split_ids["val"],
        background_frac=0.0,
        overwrite_preproc=overwrite_preproc,
        channels=channels,
        num_workers=num_workers,
    )

    return train_data, val_data


def main(config):
    train_image_folder = config.get("DEFAULT", "TrainImageFolder")
    train_label_file = config.get("DEFAULT", "TrainLabelFile")
    train_chips_path = config.get("DEFAULT", "TrainChipsPath")
    val_image_folder = config.get("DEFAULT", "ValImageFolder")
    val_label_file = config.get("DEFAULT", "ValLabelFile")
    val_chips_path = config.get("DEFAULT", "ValChipsPath")
    num_preproc_workers = config.getint("DEFAULT", "NumPreprocWorkers")
    overwrite_preproc = config.getboolean("training", "OverwritePreprocessing")
    is_distributed = config.getboolean("training", "IsDistributed")
    channels = config.get("training", "Channels").strip().split(",")
    batch_size = config.getint("training", "BatchSize")

    if num_preproc_workers > 1:
        print("Parallel mode not fully tested, reverting to serial")
        num_preproc_workers = 1

    train_data, val_data = create_datasets(
        train_data_root=train_image_folder,
        train_detect_file=train_label_file,
        train_chips_path=train_chips_path,
        val_data_root=val_image_folder,
        val_detect_file=val_label_file,
        val_chips_path=val_chips_path,
        overwrite_preproc=overwrite_preproc,
        num_workers=num_preproc_workers,
        is_distributed=is_distributed,
        channels=channels,
    )

    np.seterr(divide="ignore", invalid="ignore")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define training and validation data loaders

    if is_distributed:
        torch.distributed.init_process_group(
            backend="nccl", world_size=torch.cuda.device_count()
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
        val_sampler = torch.utils.data.SequentialSampler(val_data)

    data_loader_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("Specifying channel means...")
    if (not os.path.exists(f"{train_chips_path}/data_means.npy")) or (
        overwrite_preproc
    ):
        image_mean = [0.5] * (len(train_data.channels))
        image_std = [0.1] * (len(train_data.channels))
        np.save(f"{train_chips_path}/data_means.npy", image_mean)
        np.save(f"{train_chips_path}/data_std.npy", image_std)
        # As a start, assuming train and validation come from same distribution
        np.save(f"{val_chips_path}/data_means.npy", image_mean)
        np.save(f"{val_chips_path}/data_std.npy", image_std)
    else:
        image_mean = np.load(f"{train_chips_path}/data_means.npy")
        image_std = np.load(f"{train_chips_path}/data_std.npy")

    # instantiate model with a number of classes
    model = xView3BaselineModel(
        num_classes=len(train_data.label_map.keys()),
        num_channels=len(train_data.channels),
        image_mean=image_mean,
        image_std=image_std,
    )

    # move model to the correct device
    model.to(device)

    if is_distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[a for a in range(torch.cuda.device_count())],
            broadcast_buffers=False,
        )
        # TODO: Fix DataParallel
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for N epochs
    num_epochs = int(config["training"]["NumberEpochs"])

    for epoch in range(num_epochs):
        # train for one epoch, printing every iteration
        train_one_epoch(
            model, optimizer, data_loader_train, device, epoch, print_freq=1
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)

        checkpoint_path = f"trained_model_{epoch+1}_epochs.pth"
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete!")


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("./args.txt")

    print(f"There are {torch.cuda.device_count()} GPUs available")

    main(config)
