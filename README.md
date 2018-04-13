## Automatic catheter detection on pedeatric X-rays
This repo provides the trained model and testing code for catheter detection as described in our [paper](https://openreview.net/forum?id=By47mM_oG). 

Note that due to regulations on the patient data, we can not share the test dataset used in the paper. The test image provided in the dataset folder here was obtained by google image search with keyword "neonatal chest xray". The original image can be found [here](https://radiopaedia.org/play/11/entry/64/case/6351/studies/7717).  
<img src="sample.png" width="900px"/>

## Prerequistites
- Linux or macOS
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/xinario/catheter_detect
cd catheter_detect
```

- Download the pretrained detection model from [here](https://1drv.ms/u/s!Aj4IQl4ug0_9gj4TTqVW1JhhHG5f) (21M) and put it into the "checkpoints/catheter_detect" folder

- Run the test script
```bash
python test.py --dataroot ./datasets/pediatric_internet/ --name catheter_detect  --phase test  --loadSize 480 --sourceoftest external
```


Now you can view the result by open the html file: results/catheter_detect/test_latest/index.html

### Citations
If you find it useful and are using the code/model provided here in a publication, please cite our paper.



### Acknowlegements
[pix2pix](https://github.com/phillipi/pix2pix), [ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)
