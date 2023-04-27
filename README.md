# A Vision Transformer for Fine-grained Classification by Reducing Noise and Enhancing Discrimnative Information

PyTorch code for the paper:  *A Vision Transformer for Fine-grained Classification by Reducing Noise and Enhancing Discrimnative Information*

## Framework

![framework of ACC-ViT](./framework.png)

## Dependencies:

+ python 3.7.3
+ PyTorch 1.8.0
+ torchvision 0.9.0
+ ml_collections 0.1.0
+ numpy 1.20.1
+ pandas 1.2.3
+ scipy 1.6.2

## Usage

### 1. Download Google pre-trained ViT models

We use ViT-B_16 as the backbone. The official download link for pretrained weight [https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz](https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz).

### 2. Prepare data

In the paper, we use images from there publicly available datasets:

+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
+ [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
+ [NABirds](http://dl.allaboutbirds.org/nabirds)
+ [iNat2017](https://github.com/visipedia/inat_comp/tree/master/2017)
+ [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
+ [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

Please download them from the official websites and put them in the corresponding folders.

In order to read datasets in a unified approach, we use CSV files to read data. Please organize the train set and test set into the following forms:

|      | img_name                                                     | label |
| ---- | ------------------------------------------------------------ | ----- |
| 0    | 001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg | 1     |
| 1    | 001.Black_footed_Albatross/Black_Footed_Albatross_0074_59.jpg | 1     |
| ...  | ...                                                          | ...   |
| 5993 | 200.Common_Yellowthroat/Common_Yellowthroat_0049_190708.jpg  | 200   |

Note:

- The label of categories starts from 1 not 0.
- The order of each image does not matter.

### 3. Install required packages

Install dependencies with the following command:

```bash
pip install -r requirements.txt
```

### 4. Train

You can modify the configuration in the `config.py` file and run it directly:

```bash
python main.py
```

### 5 Validate

After model training, you will get the weight in the checkpoints folder. You can modify the configuration in  the `config.py` file and run it directly:

```bash
python val.py
```



