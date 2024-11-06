
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



### 3. Train
You can train the model using the following command:

```bash
python Train.py
```

**Note:** Please modify the dataset paths in ```Train.py``` as per your requirements.

### 4. Compute Complexity
You can test the model complexity (Params) using the following command:
```bash
python model.py
```

### 5. Test
You can test the model using the following command:

```bash
python Test.py
```

**Note:** Please modify the dataset paths in ```Test.py``` as per your requirements.

### 6. Evaluation
You can find all the evaluation metrics used in the article in the folder ```evaluation```.

**Note:** Please modify the dataset path according to your requirements.
