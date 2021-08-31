import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

from constants import FISHING, NONFISHING, PIX_TO_M
from dataloader import XView3Dataset
from utils import collate_fn, xView3BaselineModel


def center(coord):
    return (coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2))


def main(args):
    if args.scene_ids is not None:
        scene_ids = args.scene_ids.split(",")
    else:
        scene_ids = os.listdir(args.image_folder)
    checkpoint_path = args.weights
    data_root = args.image_folder
    chips_path = args.chips_path
    channels = args.channels

    # Create output directories if it does not already exist
    Path(os.path.split(args.output)[0]).mkdir(parents=True, exist_ok=True)

    if not args.device:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)

    test_data_unlabeled = XView3Dataset(
        data_root,
        None,
        "test",
        detect_file=None,
        scene_list=scene_ids,
        chips_path=chips_path,
        background_frac=0.0,
        overwrite_preproc=False,
        channels=channels,
    )

    bath_ind = test_data_unlabeled.channels.index("bathymetry")

    if not os.path.exists(f"{chips_path}/data_means.npy"):
        # image_mean, image_std = compute_channel_means(data_loader_train)
        image_mean = [0.5] * (len(test_data_unlabeled.channels))
        image_std = [0.1] * (len(test_data_unlabeled.channels))
        np.save(f"{chips_path}/data_means.npy", image_mean)
        np.save(f"{chips_path}/data_std.npy", image_std)
    else:
        image_mean = np.load(f"{chips_path}/data_means.npy")
        image_std = np.load(f"{chips_path}/data_std.npy")

    data_loader_test_unlabeled = torch.utils.data.DataLoader(
        test_data_unlabeled,
        batch_size=8,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model_eval = xView3BaselineModel(
        num_classes=len(test_data_unlabeled.label_map.keys()),
        num_channels=len(test_data_unlabeled.channels),
        image_mean=image_mean,
        image_std=image_std,
    )

    model_eval.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_eval.to(device)
    model_eval.eval()

    res = {}

    with torch.no_grad():
        for image, targets in tqdm(
            data_loader_test_unlabeled, total=len(data_loader_test_unlabeled)
        ):
            image = list(img.to(device) for img in image)
            targets = [
                {
                    k: (v.to(device) if not isinstance(v, str) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]
            if torch.cuda.is_available() and (not args.device == "cpu"):
                torch.cuda.synchronize()
            outputs = model_eval(image)
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

            for jj, (target, output) in enumerate(zip(targets, outputs)):
                output.update(
                    {
                        "pred_chip_columns": [
                            [int(np.mean([box[0], box[2]])) for box in output["boxes"]]
                            if len(output["boxes"]) > 0
                            else []
                        ]
                    }
                )
                output.update(
                    {
                        "pred_chip_rows": [
                            [int(np.mean([box[1], box[3]])) for box in output["boxes"]]
                            if len(output["boxes"]) > 0
                            else []
                        ]
                    }
                )

                # The length of the ship will be the length of any diagonal in the rectangle, which is defined by the Pythagorean
                pix_to_m = PIX_TO_M
                lengths = [
                    ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
                    for xmin, ymin, xmax, ymax in output["boxes"]
                ]
                lengths = [float(pix_to_m) * l for l in lengths]
                ratio = [
                    (xmax - xmin) / (ymax - ymin)
                    for xmin, ymin, xmax, ymax in output["boxes"]
                ]
                output.update({"lengths": lengths})

            batch_res = {
                int(target["image_id"].cpu()): {
                    "boxes": output["boxes"],
                    "lengths": output["lengths"],
                    "labels": output["labels"],
                    "scores": output["scores"],
                    "scene_id": target["scene_id"],
                    "chip_id": int(target["chip_id"].cpu()),
                    "pred_chip_columns": output["pred_chip_columns"][0],
                    "pred_chip_rows": output["pred_chip_rows"][0],
                }
                for target, output in zip(targets, outputs)
            }

            res.update(batch_res)

    df_out = pd.DataFrame(
        columns=(
            "detect_scene_row",
            "detect_scene_column",
            "scene_id",
            "is_vessel",
            "is_fishing",
            "vessel_length_m",
        )
    )

    fpath = args.output

    for inf_img_id, inf_val in tqdm(res.items()):
        for idx, box in enumerate(inf_val["boxes"]):
            label = inf_val["labels"][idx].item()
            is_fishing = label == FISHING
            is_vessel = label in [FISHING, NONFISHING]
            length = inf_val["lengths"][idx].item()
            score = inf_val["scores"][idx].item()
            scene_id = inf_val["scene_id"]
            chip_id = inf_val["chip_id"]

            # Getting chip transforms
            with open(f"{chips_path}/{scene_id}/coords.json") as fl:
                coords = json.load(fl)
            chip_offset_col, chip_offset_row = coords["offsets"][chip_id]

            # Adjusting chip pixel preds to global scene pixel preds
            chip_pred_row = inf_val["pred_chip_rows"][idx]
            chip_pred_column = inf_val["pred_chip_columns"][idx]
            scene_pred_column = chip_pred_column + chip_offset_col
            scene_pred_row = chip_pred_row + chip_offset_row

            df_out.loc[len(df_out)] = [
                scene_pred_row,
                scene_pred_column,
                scene_id,
                is_vessel,
                is_fishing,
                length,
            ]

    df_out.to_csv(fpath, index=False)
    print(f"{len(df_out)} detections found")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference on xView3 reference model."
    )

    parser.add_argument("--image_folder", help="Path to the xView3 images")
    parser.add_argument(
        "--scene_ids", help="Comma separated list of test scene IDs", default=None
    )
    parser.add_argument(
        "--chips_path",
        default=".",
        help="Path where pre-processed chips should be saved",
    )
    parser.add_argument("--weights", help="Path to trained model weights")
    parser.add_argument("--channels", nargs="+", default=["vh", "vv", "bathymetry"])
    parser.add_argument("--output", help="Path in which to output inference CSVs")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], required=False)
    parser.add_argument(
        "--num_workers", type=int, help="Number of dataloader workers", default=6
    )

    args = parser.parse_args()

    main(args)
