# xview3-public

## `/reference/train_reference.ipynb`
The above notebook in the `reference` directory can get you started training the reference model with just a few scenes downloaded from [xView3 Challenge](https://iuu.xview.us/) website.

The notebook also provides you with numerical and visual feedback, and demonstrates how the scoring metrics used for xView3 are computed.  It also provides code to visualize the performance of your model. The intent of this notebook is to (a) demonstrate an implementation for building a model on the xView3 dataset and (b) provide a set of tools that will allow you to develop your own intuition for these tasks.  It is *not* intended to recommend a particular strategy or approach!

## `environment.yml`
We strongly recommend using [Anaconda](https://www.anaconda.com/products/individual) to run the reference implementation.  To set up the python environment that includes the dependencies needed to run the reference implementation code, you can run:

```
conda env create -f environment.yml
```

You may also need to add the `xview3` environment that this command creates to your list of `jupyter` kernels.  This can be done by executing:

```
conda activate xview3
pip install ipykernel
python -m ipykernel install --user --name xview3
```

Note: if you encounter an error involving ipywidgets, this can usually be fixed by executing the following:

```
conda activate xview3
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```


## `/reference/inference.py`
This script allows you to generate predictions using the trained model weights from the reference implementation.

Example usage:
```
python ./reference/inference.py --image_folder /home/xv3data \
--scene_ids 0157baf3866b2cf9v \
--output /home/xv3data/prediction/predict.csv \
--weights ./baseline/model_weights.pth \
--chips_path /home/xv3data/chips \
--channels vh vv bathymetry
```


## `run_inference.sh`
This is the `entrypoint` to the Docker container. Please note that the shell script takes 3 positional arguments as specified in the guidelines for submission verification. The three positional arguments are:
1. `--image_folder` which is the path to the data directory.
2. `--scene_ids` which is the list of the xView3 scene identifier for which you wish to run inference.
3. `--output` which is the path to the prediction output filename in CSV format.

The other arguments uses by `inference.py`, e.g., `--channels` and `--chips_path` have been hard-coded in the shell script. You are welcome to use as many arguments for your model. However, your `entrypoint` should have exactly the three positional arguments listed above for submission verification purposes.


## `Dockerfile`
For final submission on the open-source track, top solvers will be required to provide a container that executes their model.   This is an example of a Dockerfile meeting the required specification that supports `CUDA` and `Miniconda`. To build a Docker image using the `Dockerfile` use:
```
docker build -t my-image-name .
```
where the image is named `my-image-name`.

To test your image, you can use:
```
time docker run --shm-size 16G --gpus=1 --mount type=bind,source=/home/xv3data,target=/on-docker/xv3data my-image-name /on-docker/xv3data/ 0157baf3866b2cf9v /on-docker/xv3data/prediction/prediction.csv
```
The example `docker run` utilizes a bind mount. `source=` specifies the local directory; `target=` specifies the corresponding directory in the container. The example above mounds the local `/home/xv3data` directory, allowing your container read and write access. The Docker container accesses the directory at `/on-docker/xv3data`. The example assumes that your local root xView3 data directory is `/home/xv3data` and its subdirectories containing xView3 imageries have `scene_id` names. For example, `/home/xv3data/0157baf3866b2cf9v` contains imageries for scene `0157baf3866b2cf9v`. Your Docker container will write the prediction CSV to `/on-docker/xv3data/prediction/prediction.csv` in the container which is accessible locally at `/home/xv3data/prediction/prediction.csv`.

The `time` command will give you an idea of how long it takes for your model to process a scene. Keep in mind solvers are given a maximum of 15 minutes per scene on a V100 GPU.

## Reading List
If you are unfamiliar with synthetic aperture radar (SAR), machine learning on SAR, and/or illegal, unreported, and unregulated (IUU) fishing, then this list will help you get started!

### Synthetic Aperture Radar
* [What is Synthetic Aperture Radar?](https://earthdata.nasa.gov/learn/backgrounders/what-is-sar)
* [SAR 101: An Introduction to Synthetic Aperture Radar](https://www.capellaspace.com/sar-101-an-introduction-to-synthetic-aperture-radar/)
* [Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)
* [SAR Polarimetry](https://nisar.jpl.nasa.gov/mission/get-to-know-sar/polarimetry/)

### Illegal, Unreported, and Unregulated Fishing
* [IUU – Illegal, Unreported, Unregulated Fishing](https://globalfishingwatch.org/fisheries/iuu-illegal-unreported-unregulated-fishing/)
* [Illuminating dark fishing fleets in North Korea](https://advances.sciencemag.org/content/6/30/eabb1197)
* [Illuminating the South China Sea’s Dark Fishing Fleets](https://ocean.csis.org/spotlights/illuminating-the-south-china-seas-dark-fishing-fleets/)

### Machine Learning on Synthetic Aperture Radar Imagery
* [Deep convolutional neural networks for ATR from SAR imagery](https://spie.org/Publications/Proceedings/Paper/10.1117/12.2176558?SSO=1)
* [The development of deep learning in synthetic aperture radar imagery](https://ieeexplore.ieee.org/document/7958802)
* [Synthetic Aperture Radar Ship Detection Using Capsule Networks](https://ieeexplore.ieee.org/document/8517804)