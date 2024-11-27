# CarbonBench


*A model intercomparison of neural network emulators for atmospheric transport.*


<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://twitter.com/vitusbenson" target="_blank">
    <img src="https://img.shields.io/twitter/follow/vitusbenson?style=social" alt="Twitter">
</a>
<a href="https://arxiv.org/abs/2408.11032" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2408.11032-b31b1b.svg" alt="ArXiv">
</a>

# Building CarbonBench Dataset

The script [`data/create_carbontracker_dataset.py`](data/create_carbontracker_dataset.py) can download and pre-process the CarbonTracker and ObsPack data into the CarbonBench format.

### CarbonBench LowRes
```
python data/create_carbontracker_dataset.py --save_dir path_to_store_data --gridname "latlon5.625" --vertical_levels "l10" --freq "6h"
```
### CarbonBench MidRes
```
python data/create_carbontracker_dataset.py --save_dir path_to_store_data --gridname "latlon2.8125" --vertical_levels "l20" --freq "6h"
```
### CarbonBench OrigRes
```
python data/create_carbontracker_dataset.py --save_dir path_to_store_data --gridname "latlon2x3" --vertical_levels "l34" --freq "3h"
```
### Prepare ObsPack Evaluation Data
```
python data/create_obspack_dataset.py --save_dir path_to_store_data --freq "3h"
```

# Training Models

Models can be trained with the python scripts under [`transport_models`](transport_models/)

The scripts have all the same structure and are called as follows:

```
python transport_models/carbontracker_lowres/swintransformer/swintransformer_S_p1w4_tsaf_specloss_long/train.py --data_path path_to_the_data
```

The additional flag `--rollout` initiates the n-step-ahead training schedule which should only be run after the 1-step training is completed.

The additional flag `--only_pred` runs only a forward run and evaluation and plotting routines for the best available checkpoint. 

The parent directory of each script is used as root for the experiments to be stored in. If running the scripts twice (once without and once with `--rollout`), the resulting folder structure should look like this:
```
transport_models/dataset/model/experiment
├── singlestep   		# 1-step-ahead training
|  ├── checkpoints      # model weights
|  |  ├── best.ckpt 	# best checkpoint according to validation loss
|  |  └── ...
|  ├── plots            # visualizations
|  |  ├── ckpt=best_massfixer=default/      # folder with many plots for one choice of checkpoints and massfixer
|  |  └── ...
|  ├── preds            # outputs
|  |  ├── ckpt=best_massfixer=default/      # one choice of checkpoints and massfixer
|  |  |  ├── co2_pred_rollout_QS.zarr       # full 3D field
|  |  |  └── obs_co2_pred_rollout_QS.zarr   # readout at obspack stations
|  |  └── ...
|  ├── scores           # metrics
|  |  ├── ckpt=best_massfixer=default/      # one choice of checkpoints and massfixer
|  |  |  ├── metrics.csv                    # global metrics
|  |  |  └── obs_metrics.csv                # obspack station data metrics
|  |  └── ...
|  ├── val_ds/                      # folder with validation data output during model training, for debug only
|  ├── events.out.tfevent.*         # tensorboard log data
|  └── hparams.yaml                 # hyperparameters for pytorch lightning model
├── rollout             # same as singlestep, but for the n-step-ahead training
|  ├── ...
...
```

# Installation

Make sure to install [NeuralTransport](https://github.com/vitusbenson/neural_transport), then you should be able to run the scripts.

# Cite CarbonBench

In case you use CarbonBench in your research or work, it would be highly appreciated if you include a reference to our [paper](https://arxiv.org/abs/2408.11032) in any kind of publication.

```bibtex
@article{benson2024neuraltransport,
  title = {Atmospheric Transport Modeling of CO2 with Neural Networks},
  author = {Vitus Benson, Ana Bastos, Christian Reimers, Alexander J. Winkler,
 Fanny Yang and Markus Reichstein},
  eprint={2408.11032},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2408.11032}, 
}
```

# Contact

For questions or comments regarding the usage of this repository, please use the [discussion section](https://github.com/vitusbenson/carbonbench/discussions) on Github. For bug reports and feature requests, please open an [issue](https://github.com/vitusbenson/carbonbench/issues) on GitHub.
In special cases, you can reach out to Vitus (find his email on his [website](https://vitusbenson.github.io/)).