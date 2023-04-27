import torch
import torch.nn as nn
from config import *
from datasets import read_data_set
from torchvision import transforms
from PIL import Image
from train import valid
from model import VisionTransformer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

test_transform = transforms.Compose([
    transforms.Resize(zoom_size, Image.BILINEAR),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_loader = read_data_set(test_csv, test_root, batch_size, test_transform)

model = VisionTransformer(config, input_size, num_classes=config.num_classes)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(weight_path))
model = model.cuda()
valid(model, test_loader, 'val', beta)
