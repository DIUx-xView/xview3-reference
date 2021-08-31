import datetime
import errno
import math
import os
import pickle
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

########################################################################################
################################ XVIEW 3 SPECIFIC UTILS ################################
########################################################################################


def pad(vh, rows, cols):
    """
    Pad an image to make it divisible by some block_size.
    Pad on the right and bottom edges so annotations are still usable.
    """
    r, c = vh.shape
    to_rows = math.ceil(r / rows) * rows
    to_cols = math.ceil(c / cols) * cols
    pad_rows = to_rows - r
    pad_cols = to_cols - c
    vh_pad = np.pad(
        vh, pad_width=((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
    )
    return vh_pad, pad_rows, pad_cols


def chip_sar_img(input_img, sz):
    """
    Takes a raster from xView3 as input and outputs
    a set of chips and the coordinate grid for a
    given chip size

    Args:
        input_img (numpy.array): Input image in np.array form
        sz (int): Size of chip (will be sz x sz x # of channlls)

    Returns:
        images: set of image chips
        images_grid: grid coordinates for each chip
    """
    # The input_img is presumed to already be padded
    images = view_as_blocks(input_img, (sz, sz))
    images_grid = images.reshape(
        int(input_img.shape[0] / sz), int(input_img.shape[1] / sz), sz, sz
    )
    return images, images_grid


def view_as_blocks(arr, block_size):
    """
    Break up an image into blocks and return array.
    """
    m, n = arr.shape
    M, N = block_size
    return arr.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M, N)


def find_nearest(lon, lat, x, y):
    """Find nearest row/col pixel for x/y coordinate.
    lon, lat : 2D image
    x, y : scalar coordinates
    """
    X = np.abs(lon - x)
    Y = np.abs(lat - y)
    return np.where((X == X.min()) & (Y == Y.min()))


def create_faster_rcnn_model(
    num_classes, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
):
    """
    Creates faster-rcnn model with arbitrary number of classes

    Args:
        num_classes (int): Number of classes to use
        image_mean (list, optional): Image means to use. Defaults to [0.485, 0.456, 0.406].
        image_std (list, optional): Image std to use. Defaults to [0.229, 0.224, 0.225].

    Returns:
        torch.Module : Pytorch model module
    """

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, image_mean=image_mean, image_std=image_std
    )

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # multi-class problem (with background)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class xView3BaselineModel(nn.Module):
    """
    Baseline model class for xView3 reference implementation.
    Wraps torchvision faster-rcnn, updates initial layer to handle
    man arbitrary number of input channels.
    """

    def __init__(self, num_classes, num_channels, image_mean, image_std):
        super(xView3BaselineModel, self).__init__()
        self.faster_rcnn = create_faster_rcnn_model(
            num_classes, image_mean=image_mean, image_std=image_std
        )
        print(f"Using {num_channels} channels for input layer...")
        if num_channels != 3:
            # Adjusting initial layer to handle arbitrary number of inputchannels
            self.faster_rcnn.backbone.body.conv1 = nn.Conv2d(
                num_channels,
                self.faster_rcnn.backbone.body.conv1.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, *input, **kwargs):
        out = self.faster_rcnn.forward(*input, **kwargs)
        return out


def rasterio_transform_to_gdal_transform(rio_transform):
    """
    Converts rasterio transform to gdal transform
    """
    return (
        rio_transform[2],
        rio_transform[0],
        rio_transform[1],
        rio_transform[5],
        rio_transform[3],
        rio_transform[4],
    )


def coord_to_pixel(x, y, transform, err=0.1):
    """From geospatial coordinate (x,y) to image pixel (row,col).
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to
    calculate the pixel location of a geospatial coordinate.
    x/y and transform must be in the same georeference.
    err is truncation error to return integer indices.
        if err=None, return pixel coords as floats.
    """
    if np.ndim(x) == 0:
        x, y = np.array([x]), np.array([y])
    col = (x - transform[0]) / transform[1]
    row = (y - transform[3]) / transform[5]
    if err:
        row = (row + err).astype(int)
        col = (col + err).astype(int)
    return row.item(), col.item()


def pixel_to_coord(row, col, transform):
    """From image pixel (i,j) to geospatial coordinate (x,y)."""
    if np.ndim(row) == 0:
        row, col = np.array([row]), np.array([col])
    x = transform[0] + col * transform[1]
    y = transform[3] + row * transform[5]
    return x.item(), y.item()


########################################################################################
############################ PYTORCH OBJECT DETECTION UTILS ############################
########################################################################################

"""
Source: https://github.com/pytorch/vision/blob/master/references/detection/utils.py
"""


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
