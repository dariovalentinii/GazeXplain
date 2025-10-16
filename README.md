# GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths

<p align="center">
  <a href="https://scholar.google.com/citations?hl=en&user=gjH1-CcAAAAJ">Xianyu Chen</a>,
  <a href="https://scholar.google.com/citations?user=JbLTK4AAAAAJ&hl=en">Ming Jiang</a>, 
  <a href="https://scholar.google.com/citations?user=JO40q04AAAAJ&hl=en">Qi (Catherine) Zhao</a>
  <br><br>
  University of Minnesota<br>
  <br>
<i><strong><a href='https://eccv2024.ecva.net/' target='_blank'>ECCV 2024 Oral</a></strong></i>
</p>

This code implements the prediction of visual scanpath along with its corresponding natural language explanations in three different tasks (3 different datasets) with two different architecture:

- Free-viewing: the prediction of scanpath for looking at some salient or important object in the given image. (OSIE)
- Visual Question Answering:  the prediction of scanpath during human performing general tasks, e.g., visual question answering, to reflect their attending and reasoning processes. (AiR-D)
- Visual search: the prediction of scanpath during the search of the given target object to reflect the goal-directed behavior under target present and absent conditions. (COCO-Search18 Target-Present and Target-Absent)

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2408.02788)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/chenxy99/GazeXplain)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=kTv-dDUBOd8)

:fire: News <a name="news"></a>
------------------
- `[2024/08]` 🎉 Our paper is selected as an oral presentation.
- `[2024/07]` GazeXplain code and [datasets](#datasets) initially released.

:mega: Overview
------------------
![overall_structure](./asset/teaser.png)
We introduce ``GazeXplain``, a novel scanpath explanation task to understand human visual attention. We provide ground-truth explanations on various eye-tracking datasets and develop a model architecture for predicting scanpaths and generating natural language explanations.
This example reveals how observers strategically investigate a scene to find out if the person is walking on the sidewalk. Fixations (circles) start centrally, locating a driving car, then shifting to the sidewalk to find the person, and finally looking down to confirm if the person is walking. By annotating observers' scanpaths with detailed explanations, we enable a deeper understanding of the what and why behind fixations, providing insights into human decision-making and task performance.


:bowing_man: Disclaimer
------------------
For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package. 
We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics mentioned in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). 
More specific, we adopt the evaluation metrics provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer), respectively.
Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate our pipeline.

:white_check_mark: Requirements
------------------

- Python 3.10
- PyTorch 2.1.2 (along with torchvision)

- We also provide the conda environment ``environment.yml``, you can directly run

```bash
$ conda env create -f environment.yml
$ conda activate gazexplain-mac
```

The provided `environment.yml` installs a CPU-only stack suitable for running inference on Intel macOS systems. After creating the environment, activate it with the second command before running any of the project scripts. When finished, deactivate the session with `conda deactivate`, and remove it entirely with `conda env remove -n gazexplain-mac` if you no longer need it.

:bookmark_tabs: Datasets <a name="datasets"></a>
------------------

Our GazeXplain dataset is released! You can download the dataset from [`Link`](https://drive.google.com/drive/folders/13-0j4wkCmab_8Uge30bwCd-vJ1k6gxzO?usp=sharing). 
This dataset contains the explanations of visual scanpaths in three different scanpath datasets (OSIE, AiR-D, COCO-Search18).

:computer: Preprocess
------------------

To process the data, you can follow the instructions provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer).
For handling the SS cluster, you can refer to [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer) and [`Target-absent-Human-Attention`](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention).
More specifically, you can run the following scripts to process the data.

```bash
$ python ./src/preprocess/${dataset}/preprocess_fixations.py
```

```bash
$ python ./src/preprocess/${dataset}/feature_extractor.py
```


We structure `<dataset_root>` as follows

#### OSIE `clusters.npy` generation workflow

`clusters.npy` stores the MeanShift models and discrete fixation strings required by the scanpath similarity metrics. The evaluator searches this file for keys that match the `<split>-<image_id>` pattern (for example, `test-1014`). Without the file, inference fails when metrics are computed.

1. **Prepare the processed directory** – ensure the following files live under `<dataset_root>/OSIE/processed/`:

   - `fixations/` – raw JSON files for each split (`osie_fixations_train.json`, `osie_fixations_validation.json`, `osie_fixations_test.json`).
   - `fixations.json` – a merged view containing every scanpath entry. Create/update it with:

     ```bash
     python src/preprocess/OSIE/preprocess_fixations.py \
       --dataset_dir <dataset_root>/OSIE
     ```

     Run this from an activated conda environment so NumPy, SciPy, and scikit-learn are available.

   - `explanation.json` – textual annotations shipped with the dataset.

2. **Run the clustering script** – `src/preprocess/OSIE/proprecess_OSIE_SScluster.py` is a standalone script. It reads `fixations.json`, rescales coordinates to `384×512`, fits MeanShift, and writes `clusters.npy`. Before executing, adjust the hard-coded roots at the top of the file (or create symbolic links) so that `fixation_root`/`processed_root` both point to `<dataset_root>/OSIE/processed/` on your machine. Then launch:

   ```bash
   python src/preprocess/OSIE/proprecess_OSIE_SScluster.py
   ```

   The script prints progress while iterating over images and produces `<dataset_root>/OSIE/processed/clusters.npy`.

   > **Tip:** if you prefer not to edit the script, temporarily create `/home/OSIE/processed` and symlink it to your processed folder so the default paths resolve:
   > ```bash
   > sudo mkdir -p /home/OSIE
   > sudo ln -s <dataset_root>/OSIE/processed /home/OSIE/processed
   > ```
   > Remember to remove the symlink when you finish preprocessing.

3. **Validate the output** – load the file in a Python shell to verify the expected keys exist:

   ```python
   import numpy as np
   clusters = np.load('<dataset_root>/OSIE/processed/clusters.npy', allow_pickle=True).item()
   print(len(clusters))            # number of split-image entries
   print(list(clusters)[:5])       # sample keys such as 'test-1014'
   ```

Once `clusters.npy` is present alongside `fixations.json`, the evaluation script can compute scanpath metrics for OSIE samples during inference.

:runner: Training your own network on ALL the datasets
------------------

We set all the corresponding hyper-parameters in ``opt.py``. 

The `train_explanation_alignment.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./runs/`). You can also set the other hyper-parameters in `opt.py` or define them in the `bash/train.sh`.

- `--datasets` Folder to the dataset, e.g., `<dataset_root>`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning at the predefined epoch.

You can also use the following commands to train your own network. Then you can run the following commands to evaluate the performance of your trained model on test split.
```bash
$ sh bash/train.sh
```

:bullettrain_front:	Evaluate on test split
------------------
For inference, we provide the [`pretrained model`](https://drive.google.com/file/d/10WfTJOeF4LjsmILUTb0Z0tVOgdu0P21Q/view?usp=sharing), and you can directly run the following command to evaluate the performance of the pretrained model on test split.
```bash
$ sh bash/test.sh
```

If you prefer to invoke the evaluation script manually, ensure that the
`accelerate` CLI comes from the active conda environment and run:

```bash
$ accelerate launch --cpu --config_file src/config.yaml \
    src/test_explanation_alignment.py --split test --test_batch 1 \
    --dataset_dir <dataset_root> --datasets OSIE
```

In case your shell cannot find the console script or you receive an error like
```
No module named accelerate.__main__; 'accelerate' is a package and cannot be directly executed
```
use the module path provided by the package instead:

```bash
$ python -m accelerate.commands.launch --cpu --config_file src/config.yaml \
    src/test_explanation_alignment.py --split test --test_batch 1 \
    --dataset_dir <dataset_root> --datasets OSIE
```

### Troubleshooting common macOS runtime errors

When creating a minimal conda environment you may have to add a few packages
manually as you encounter import errors:

- **Missing `cv2`** – install OpenCV inside the active environment:
  ```bash
  (gazexplain-mac) $ conda install -c conda-forge opencv
  # or, if you prefer pip:
  (gazexplain-mac) $ python -m pip install opencv-python
  ```
- **`RequestsDependencyWarning` about `chardet`/`charset_normalizer`** – add a
  charset detector so the `requests` library can guess encodings:
  ```bash
  (gazexplain-mac) $ python -m pip install charset-normalizer
  ```
- **Accelerate/Numpy compatibility** – the CPU wheels used here expect NumPy
  1.26.x. If you see `module 'numpy' has no attribute '_core'`, reinstall the
  compatible stack:
  ```bash
  (gazexplain-mac) $ python -m pip install 'numpy<2' 'accelerate==0.27.0'
  ```
  and then rerun the launch command from the same shell.
**`RuntimeError: Device index must not be negative` appears even with**
`--eval_repeat_num 1`
--------------------------------------------------------------

The sampling routine invoked during evaluation (`self.sampling.random_sample`
inside `src/lib/models/gazeformer_explanation_alignment.py`) creates Gaussian
noise with `torch.randn(...).to(log_normal_mu.get_device())`. On CPU tensors
`Tensor.get_device()` returns `-1`, so PyTorch raises `RuntimeError: Device
index must not be negative` every time inference is launched with
`accelerate --cpu`—regardless of the `--eval_repeat_num` value.

At the moment the repository does not provide a CPU-only execution path for
sampling. Resolving the error therefore requires one of the following:

1. **Run inference on hardware with a GPU/MPS device.** Activate the same
   conda environment on a machine that exposes CUDA or Apple Metal, copy your
   `runs/` and processed dataset folders across, and invoke the launch command
   without the `--cpu` flag (Accelerate will select the GPU automatically).

2. **Patch the local code to support CPU tensors.** If modifying files is an
   option, replace the `torch.randn(...).to(log_normal_mu.get_device())` call
   in `src/lib/models/sample/sampling.py` with `torch.randn(...,
   device=log_normal_mu.device)` or move the model to GPU after loading the
   checkpoint. Either change prevents the negative-device lookup.

For users who must remain on Intel-only macOS hardware and cannot adjust the
code, the practical workaround is to offload inference to a remote GPU (e.g.,
Google Colab, an on-premises workstation, or a rented cloud VM) by copying the
dataset and `runs/ALL_runX_baseline` directory to that system.
  and dataset.
  ```bash
  (gazexplain-mac) $ accelerate launch --config_file src/config.yaml \
      src/test_explanation_alignment.py --split test --test_batch 1 \
      --dataset_dir <dataset_root> --datasets OSIE --eval_repeat_num 1
  ```
  If you must stay on CPU-only hardware, patch `sampling.py` so that it keeps
  tensors on the CPU (replace `.to(log_normal_mu.get_device())` with
  `.to(log_normal_mu.device)`). This requires modifying the source code, which
  is why we recommend borrowing GPU resources instead when code changes are not
  an option.

### Fixing `RuntimeError: stack expects a non-empty TensorList`

This happens when `--eval_repeat_num` is set to `0`. The inference loop in
`gazeformer_explanation_alignment.py` accumulates scanpath samples inside a
Python list and calls `torch.stack` afterwards; with zero repeats the list stays
empty and the call fails. Always keep `--eval_repeat_num ≥ 1` unless you are
prepared to edit the model code to short-circuit the stacking step.

### Fixing `mat1 and mat2 shapes cannot be multiplied`

This runtime exception indicates that the image features loaded from
`<dataset_root>/OSIE/image_features/*.pth` do not have the expected channel
dimension of 2048. It typically appears after extracting features with a script
that stores 256-dimensional FPN maps. Remove any incorrectly generated files
and recreate them with the reference extractor:

```bash
(gazexplain-mac) $ find <dataset_root>/OSIE/image_features -name '*.pth' -delete
(gazexplain-mac) $ python - <<'PY'
import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image

class ResNetCOCO(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.resnet = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        b, c, h, w = x.shape
        return x.view(b, c, h * w).permute(0, 2, 1).contiguous()

dataset_path = os.path.expanduser('<dataset_root>/OSIE')
stimuli_dir = os.path.join(dataset_path, 'stimuli')
target_dir = os.path.join(dataset_path, 'image_features')
os.makedirs(target_dir, exist_ok=True)

resize = T.Resize((384 * 2, 512 * 2))
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
backbone = ResNetCOCO(device="cpu").eval()

for fname in sorted(os.listdir(stimuli_dir)):
    if not fname.lower().endswith('.jpg'):
        continue
    image = Image.open(os.path.join(stimuli_dir, fname)).convert('RGB')
    tensor = normalize(resize(T.functional.to_tensor(image))).unsqueeze(0)
    features = backbone(tensor).squeeze(0).cpu()
    torch.save(features, os.path.join(target_dir, fname.replace('.jpg', '.pth')))
PY
```

After regeneration, spot-check a file to confirm its shape (`torch.load(...).shape`
should be `(H×W, 2048)` with `H×W = 96×128 = 12288` for the resized resolution).


:black_nib: Citation
------------------
If you use our code or data, please cite our paper:
```text
@inproceedings{xianyu:2024:gazexplain,
    Author         = {Xianyu Chen and Ming Jiang and Qi Zhao},
    Title          = {GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths},
    booktitle      = {Proceedings of the European Conference on Computer Vision (ECCV)},
    Year           = {2024}
}
```