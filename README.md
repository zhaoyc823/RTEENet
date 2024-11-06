
# RTEE-Net: Real-time Efficient Image Enhancement in Low-light Condition with Novel Supervised Deep Learning Pipeline
## Description
This is the PyTorch version of RTEE-Net.

## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n RTEE-Net python=3.9 -y
conda activate RTEE-Net
```
- Install Dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

### 2. Prepare Datasets
Download the LOLv1 and LOLv2 datasets:

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

<details>
  <summary>
  <b>Datasets should be organized as follows:</b>
  </summary>

  ```
    |--LOL_v1   
    |    |--Train
    |    |    |--low
    |    |    |  ...
    |    |    |--high
    |    |    |  ...
    |    |--Test
    |    |    |--low
    |    |    |  ...
    |    |    |--high
    |    |    |  ...
    |--LOL_v2_real
    |    |--Train
    |    |    |--low
    |    |    |  ...
    |    |    |--high
    |    |    |  ...
    |    |--Test
    |    |    |--low
    |    |    |  ...
    |    |    |--high
    |    |    |  ...
    |--LOL_v2_Syn
    |    |--Train
    |    |    |--low
    |    |    |  ...
    |    |    |--high
    |    |    |  ...
    |    |--Test
    |    |    |--low
    |    |    |  ...
    |    |    |--high
    |    |    |  ...
  ```

</details>

**Note:** ```data``` directory should be placed under the ```PyTorch``` implementation folder.

### 3. Test
You can test the model using the following commands. Pre-trained weights are available at [Google Drive](https://drive.google.com/file/d/1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1/view?usp=sharing). GT Mean evaluation is enabled by default and can be deactivated by setting the boolean flag ```gt_mean=False``` in the ```compute_psnr()``` method under the ```test.py``` file.

```bash
python test.py
```

**Note:** Please modify the dataset paths in ```test.py``` as per your requirements.

### 4. Compute Complexity
You can test the model complexity (FLOPS/Params) using the following command:
```bash
# To run FLOPS check with default (1,256,256,3)
python macs.py
```

### 5. Train
You can train the model using the following command:

```bash
python train.py
```

**Note:** Please modify the dataset paths in ```train.py``` as per your requirements.
