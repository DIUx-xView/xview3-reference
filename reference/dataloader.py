import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import ray
import torch
from rasterio.enums import Resampling

from constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL
from utils import chip_sar_img, pad


def get_grid_coords(padded_img, chips, grids):
    """
    Obtain grid coordinates for chip origins from padded images
    """
    chip_size = chips[0].shape[0]
    grid_coords_y = np.linspace(0, padded_img.shape[0] - chip_size, grids.shape[0])
    grid_coords_x = np.linspace(0, padded_img.shape[1] - chip_size, grids.shape[1])
    grid_coords = [(int(x), int(y)) for y in grid_coords_y for x in grid_coords_x]
    return grid_coords


def scene_pixels_to_chip_pixels(chips, grid_coords, scene_rows, scene_cols):
    """
    Convert scene-level pixel indices for detections to chip-level indices
    """
    chip_size = chips[0].shape[0]
    chip_rows = (scene_rows) % (chip_size)
    chip_cols = (scene_cols) % (chip_size)
    chip_ind_row = (scene_rows) // chip_size * chip_size
    chip_ind_col = (scene_cols) // chip_size * chip_size
    chip_index = [grid_coords.index((c, r)) for c, r in zip(chip_ind_col, chip_ind_row)]
    return chip_rows, chip_cols, chip_index, grid_coords


def process_scene(
    scene_id,
    detections,
    channels,
    chip_size,
    chips_path,
    overwrite_preproc,
    root,
    split,
    index,
):
    """
    Preprocess scene by loading images, chipping them,
    saving chips and grid coordinates, converting scene-level to
    chip-level detections, and returning a dataframe with
    information required for training.
    """

    pixel_detections = pd.DataFrame()

    # If detections file exists, load chip-level annotations for the scene;
    # otherwise, assume this is for inference

    if detections is not None:
        scene_detects = detections[detections["scene_id"] == scene_id]
        scene_detect_num = len(scene_detects)
        print(
            f"Detections expected for scene # {index} ({scene_id}): {scene_detect_num}"
        )
        # Loading up detections file
        if not overwrite_preproc and os.path.exists(
            f"{chips_path}/{split}_chip_annotations.csv"
        ):
            pixel_detections = pd.read_csv(f"{chips_path}/{split}_chip_annotations.csv")
            pixel_detections = pixel_detections[
                (pixel_detections["scene_id"] == scene_id)
            ]
    else:
        scene_detect_num = 0
        print(f"No detection file, only chipping for inference")

    # Logic for overwriting existing detectinos file
    if overwrite_preproc and os.path.exists(
        f"{chips_path}/{split}_chip_annotations.csv"
    ):
        os.remove(f"{chips_path}/{split}_chip_annotations.csv")

    # Use xView3 data file structure to define files to load up
    # for each possible channel
    files = {}
    files["vh"] = os.path.join(root, f"{scene_id}", "VH_dB.tif")
    files["vv"] = Path(files["vh"]).parent / "VV_dB.tif"
    files["bathymetry"] = Path(files["vh"]).parent / "bathymetry.tif"
    files["wind_speed"] = Path(files["vh"]).parent / "owiWindSpeed.tif"
    files["wind_direction"] = Path(files["vh"]).parent / "owiWindDirection.tif"
    files["wind_quality"] = Path(files["vh"]).parent / "owiWindQuality.tif"
    files["mask"] = Path(files["vh"]).parent / "owiMask.tif"

    imgs, chips, grids = {}, {}, {}

    # For each channel, if it is already chipped, do not re-chip
    for fl in channels:
        temp_folder = Path(chips_path) / scene_id / fl
        if os.path.exists(temp_folder) and (not overwrite_preproc):
            print(f"Using existing preprocessed {fl} data for scene {scene_id}")
            continue
        else:
            os.makedirs(temp_folder, exist_ok=True)
        src = rasterio.open(files[fl])
        imgs[fl] = src.read(1)

        # If not same size as first channel, resample before chipping
        # to ensure chips from different channels are co-registered
        if not imgs[fl].shape == imgs[channels[0]].shape:
            imgs[fl] = src.read(
                out_shape=(
                    imgs[channels[0]].shape[0],
                    imgs[channels[0]].shape[1],
                ),
                resampling=Resampling.bilinear,
            ).squeeze()
        try:
            assert imgs[fl].shape == imgs[channels[0]].shape
        except AssertionError as e:
            print(f"imgs[fl].shape = {imgs[fl].shape}")
            print(f"imgs[channels[0]].shape = {imgs[channels[0]].shape}")
            raise AssertionError()

        # Pad the raster to be a multiple of the chip size
        padded_img, _, _ = pad(imgs[fl], chip_size, chip_size)

        # Get image chips and grids
        chips[fl], grids[fl] = chip_sar_img(padded_img, chip_size)

        # Saving chips
        for i in range(len(chips[fl])):
            chip = chips[fl][i]
            np.save(f"{temp_folder}/{i}_{fl}.npy", chip)

        if fl == channels[0]:
            # Getting grid coordinates
            grid_coords = get_grid_coords(padded_img, chips[fl], grids[fl])

            # Saving offsets for each chip; these offsets are alsp needed to convert
            # chip-level predictions to scene-level predictions at
            # inference time
            write_object = {
                "offsets": grid_coords,
            }
            json.dump(
                write_object, open(Path(chips_path) / scene_id / "coords.json", "w")
            )

            if detections is not None:
                print("Getting detections...")
                # Get pixel values for detections in scene
                (scene_detects["scene_rows"], scene_detects["scene_cols"],) = (
                    scene_detects["detect_scene_row"],
                    scene_detects["detect_scene_column"],
                )

                # Convert scene-level detection coordinates to chip-level
                # detection coordinates
                (
                    chip_rows,
                    chip_cols,
                    chip_indices,
                    grid_coords,
                ) = scene_pixels_to_chip_pixels(
                    chips[fl],
                    grid_coords,
                    np.array(scene_detects["scene_rows"]),
                    np.array(scene_detects["scene_cols"]),
                )

                scene_detects["rows"] = chip_rows
                scene_detects["columns"] = chip_cols
                scene_detects["chip_index"] = chip_indices

                pixel_detections = scene_detects

                # Append to annotations file
                if not os.path.exists(f"{chips_path}/{split}_chip_annotations.csv"):
                    pixel_detections.to_csv(
                        f"{chips_path}/{split}_chip_annotations.csv",
                        mode="w",
                        header=True,
                    )
                else:
                    pixel_detections.to_csv(
                        f"{chips_path}/{split}_chip_annotations.csv",
                        mode="a",
                        header=False,
                    )

    # Print number of detections per scene; make sure it aligns with
    # number expected
    if detections is not None:
        if not overwrite_preproc:
            chip_detect_num = len(
                pixel_detections[
                    (pixel_detections["scene_id"] == scene_id)
                    & (pixel_detections["vessel_class"] != BACKGROUND)
                ]
            )
        else:
            chip_detect_num = len(pixel_detections)

        print(
            f"Detections recovered in chips for scene {scene_id}: {chip_detect_num} \n"
        )

    return pixel_detections


class XView3Dataset(object):
    """
    Pytorch dataset for training Faster R-CNN with
    xView3 data
    """

    def __init__(
        self,
        root,
        transforms,
        split,
        detect_file=None,
        scene_list=None,
        chips_path=".",
        channels=["vh", "vv", "wind_direction"],
        chip_size=800,
        overwrite_preproc=False,
        bbox_size=5,
        background_frac=None,
        background_min=3,
        ais_only=True,
        num_workers=1,
        min_max_norm=True,
    ):

        self.root = root
        self.split = split
        self.bbox_size = bbox_size
        self.background_frac = background_frac
        self.background_min = background_min
        self.chips_path = chips_path
        self.transforms = transforms
        self.channels = channels
        self.chip_size = chip_size
        self.overwrite_preproc = overwrite_preproc
        self.ais_only = ais_only
        self.num_workers = num_workers
        self.label_map = self.get_label_map()
        self.min_max_norm = min_max_norm
        self.coords = {}

        # Getting image lst
        if not scene_list:
            self.scenes = [
                a.strip("\n").strip("/").split("/")[-1][:67] for a in os.listdir(root)
            ]
        else:
            self.scenes = scene_list

        # Get all detections; convert label schema from xView3 label file
        # to a three-class label schema that can be used by Faster R-CNN
        if detect_file:
            self.detections = pd.read_csv(detect_file, low_memory=False)
            vessel_class = []
            for ii, row in self.detections.iterrows():
                if row.is_vessel and row.is_fishing:
                    vessel_class.append(FISHING)
                elif row.is_vessel and not row.is_fishing:
                    vessel_class.append(NONFISHING)
                elif not row.is_vessel:
                    vessel_class.append(NONVESSEL)
            self.detections["vessel_class"] = vessel_class
            # Assuming we're only using examples with vessel class for this
            # training procedure
            if self.ais_only:
                self.detections = self.detections.dropna(subset=["vessel_class"])
        else:
            self.detections = None

        # Get chip-level detection coordinates
        self.pixel_detections = self.chip_and_get_pixel_detections()

        # Add background chips for negative sampling
        if self.background_frac and (self.detections is not None):
            print("Adding background chips...")
            self.add_background_chips()

        # Write annotations to file
        if self.overwrite_preproc or not os.path.exists(
            f"{self.chips_path}/{self.split}_chip_annotations.csv"
        ):
            print("Writing chip annotations to file...")
            self.pixel_detections.to_csv(
                f"{self.chips_path}/{self.split}_chip_annotations.csv", index=False
            )
        # Get chip indices for each scene
        if self.detections is not None:
            self.chip_indices = list(
                set(
                    zip(
                        self.pixel_detections.scene_id, self.pixel_detections.chip_index
                    )
                )
            )
        else:
            self.chip_indices = []
            for scene_id in self.scenes:
                chip_num = self.get_chip_number(scene_id)
                self.chip_indices += [(scene_id, a) for a in range(chip_num)]

        print(f"Number of Unique Chips: {len(self.chip_indices)}")
        print("Initialization complete")

    @staticmethod
    def get_label_map():
        """
        Not currently used, but useful for laying out how the
        fishing vs. non-fishing labels can be separated.
        """
        background_labels = ["background"]
        non_fishing_labels = [
            "cargo",
            "cargo_or_tanker",
            "tanker",
            "tug",
            "supply_vessel",
            "non_fishing",
            "bunker_or_tanker",
            "dredge_non_fishing",
            "other_not_fishing",
            "cargo_or_reefer",
            "well_boat",
            "container_reefer",
            "bunker",
            "specialized_reefer",
            "reefer",
            "submarine",
        ]
        fishing_labels = [
            "other_purse_seines",
            "trawlers",
            "fishing",
            "dredge_fishing",
            "tuna_purse_seines",
            "purse_seines",
            "set_longlines",
            "pots_and_traps",
            "set_gillnets",
            "drifting_longlines",
            "pole_and_line",
            "other_seines",
            "trollers",
            "seiners",
        ]
        personnel_labels = [
            "passenger",
            "patrol_vessel",
            "seismic_vessel",
            "dive_vessel",
            "troller",
            "research",
        ]
        other_labels = ["fixed_gear", "helicopter", "gear"]

        label_map = {a: NONFISHING for a in non_fishing_labels}
        label_map.update({a: FISHING for a in fishing_labels})
        label_map.update({a: NONVESSEL for a in personnel_labels})
        label_map.update({a: NONVESSEL for a in other_labels})

        # Treat background as a separate class for classifier
        # to support adding empty chips for training
        label_map.update({a: BACKGROUND for a in background_labels})

        return label_map

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        # Load and condition image chip data
        scene_id, chip_index = self.chip_indices[idx]
        data = {}
        for fl in self.channels:
            pth = f"{self.chips_path}/{scene_id}/{fl}/{int(chip_index)}_{fl}.npy"
            data[fl] = np.load(pth)
            if fl == "wind_direction":
                data[fl][data[fl] < 0] = np.random.randint(0, 360, size=1)
                data[fl][data[fl] > 360] = np.random.randint(0, 360, size=1)
                data[fl] = data[fl] - 180
            if fl == "wind_speed":
                data[fl][data[fl] < 0] = 0
                data[fl][data[fl] > 100] = 100
            if fl in ["vh", "vv"]:
                data[fl][data[fl] < -50] = -50
            if self.min_max_norm:
                # Puts values b/t 0 and 1, as expected by Faster-RCNN implementation
                data[fl] = (data[fl] - np.min(data[fl])) / (
                    np.max(data[fl]) - np.min(data[fl])
                )

        # Stacking channels to create multi-band image chip
        img = torch.tensor(np.array([data[fl] for fl in self.channels]))

        # Get label information if it exists
        if self.detections is not None:
            detects = self.pixel_detections[
                (self.pixel_detections["scene_id"] == scene_id)
                & (self.pixel_detections["chip_index"] == chip_index)
            ]

            num_objs = len(detects)
            boxes = []
            class_labels = []
            length_labels = []
            if (num_objs == 1) and (detects.iloc[0].vessel_class == BACKGROUND):
                # Adding boxes in correct format for background chips
                # See https://github.com/pytorch/vision/pull/1911
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros((1,), dtype=torch.int64)
                length_labels = torch.zeros((1,), dtype=torch.float32)

            else:

                for i in range(num_objs):
                    # Setting uniform bounding boxes around each detection
                    # so Faster R-CNN can train; note that this is not a good
                    # way to build a model that will estimate vessel length!
                    detect = detects.iloc[i]
                    xmin = detect.columns - self.bbox_size
                    xmax = detect.columns + self.bbox_size
                    ymin = detect.rows - self.bbox_size
                    ymax = detect.rows + self.bbox_size
                    boxes.append([xmin, ymin, xmax, ymax])
                    class_labels.append(detect.vessel_class)
                    length_labels.append(detect.vessel_length_m)

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
                length_labels = torch.as_tensor(length_labels, dtype=torch.float32)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # Return dummy values for inference
            boxes = torch.tensor([])
            class_labels = torch.tensor(-1)
            length_labels = torch.tensor(-1)
            area = torch.tensor(-1)
            num_objs = 0

        # Create target dictionary in expected format for Faster R-CNN
        target = {}
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["length_labels"] = length_labels
        target["scene_id"] = scene_id
        target["chip_id"] = torch.tensor(chip_index)
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            # Currently not used; applies image transforms
            img, target = self.transforms(img, target)

        return img.float(), target

    def get_chip_number(self, scene_id):
        """
        Get number of chips using first channel
        """
        return len(glob.glob(f"{self.chips_path}/{scene_id}/{self.channels[0]}/*.npy"))

    def add_background_chips(self):
        """
        Add background chips with no detections
        """
        for scene_id in self.scenes:
            # getting chip number for scene
            num_chips = self.get_chip_number(scene_id)

            # getting chips that have detections
            scene_detect_chips = (
                self.pixel_detections[self.pixel_detections["scene_id"] == scene_id][
                    "chip_index"
                ]
                .astype(int)
                .tolist()
            )

            # getting chips that do not have any detections
            scene_background_chips = [
                a for a in range(num_chips) if a not in list(set(scene_detect_chips))
            ]

            # computing the number of chips required
            num_background = int(
                self.background_frac * max(len(scene_detect_chips), self.background_min)
            )

            # obtaining a random set of chips without detections as background;
            # adding rows to a dataframe that will be appended to the dataset's
            # pixel_detections field
            np.random.seed(seed=0)
            chip_nums = np.random.choice(
                scene_background_chips, size=num_background, replace=False
            )
            rows = []
            cols = [
                "index",
                "detect_lat",
                "detect_lon",
                "vessel_length_m",
                "source",
                "detect_scene_row",
                "detect_scene_column",
                "is_vessel",
                "is_fishing",
                "distance_from_shore_km",
                "scene_id",
                "confidence",
                "top",
                "left",
                "bottom",
                "right",
                "detect_id",
                "vessel_class",
                "scene_rows",
                "scene_cols",
                "rows",
                "columns",
                "chip_index",
            ]
            for ii in range(num_background):
                row = [
                    -1,  #'index'
                    -1,  #'detect_lat'
                    -1,  # 'detect_lon'
                    -1,  #'vessel_length_m'
                    "background",  # source
                    -1,  #'detect_scene_row',
                    -1,  #'detect_scene_column',
                    -1,  #'is_vessel'
                    -1,  #'is_fishing',
                    -1,  #'distance_from_shore_km',
                    scene_id,  #'scene_id',
                    -1,  #'confidence',
                    -1,  # top
                    -1,  # left
                    -1,  # bottom
                    -1,  # right
                    -1,  #'detect_id',
                    BACKGROUND,  #'vessel_class',
                    -1,  #'scene_rows',
                    -1,  #'scene_cols',
                    -1,  #'rows',
                    -1,  #'columns',
                    chip_nums[ii],  # chip_index
                ]

                rows.append(row)
            # Background chips for this scene to dataframe
            df_background = pd.DataFrame(rows, columns=cols)
            # Append background chips for this scene to dataset-level detections
            # dataframe
            self.pixel_detections = pd.concat((self.pixel_detections, df_background))

    def chip_and_get_pixel_detections(self):
        """
        Preprocess all scenes to chip xView3 dataset images
        and create global detections dataframe and .csv file.

        NOTE: parallelization does not currently work, but can
        be done in this style if the process_scene function
        were to be appropriately modified
        """

        start = time.time()
        if self.num_workers > 1:
            ray.init()
            # Use ray remote for parallelized preprocessing
            remote_process_scene = ray.remote(process_scene)

            jobs = []
            for jj, scene_id in enumerate(self.scenes):
                jobs.append(
                    remote_process_scene.remote(
                        scene_id,
                        self.detections,
                        self.channels,
                        self.chip_size,
                        self.chips_path,
                        self.overwrite_preproc,
                        self.root,
                        self.split,
                        jj,
                    )
                )

            chip_detects = ray.get(jobs)
            pixel_detections = pd.concat(chip_detects)

        else:
            # Otherwise, just use a for loop
            chip_detects = []

            for jj, scene_id in enumerate(self.scenes):
                print(f"Processing scene {jj} of {len(self.scenes)}...")
                chip_detects.append(
                    process_scene(
                        scene_id,
                        self.detections,
                        self.channels,
                        self.chip_size,
                        self.chips_path,
                        self.overwrite_preproc,
                        self.root,
                        self.split,
                        jj,
                    )
                )

            pixel_detections = pd.concat(chip_detects).reset_index()

        el = time.time() - start
        print(f"Elapsed Time: {np.round(el/60, 2)} Minutes")

        return pixel_detections
