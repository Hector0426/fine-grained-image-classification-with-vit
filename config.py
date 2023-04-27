import ml_collections


def get_b16_config():
    """
    about model
    :return:
    """
    c = ml_collections.ConfigDict()
    c.patches = ml_collections.ConfigDict({'size': (16, 16)})
    c.split = 'non-overlap'
    c.slide_step = 12
    c.hidden_size = 768
    c.transformer = ml_collections.ConfigDict()
    c.transformer.mlp_dim = 3072
    c.transformer.num_heads = 12
    c.transformer.num_layers = 12
    c.transformer.attention_dropout_rate = 0.0
    c.transformer.dropout_rate = 0.1
    c.classifier = 'token'
    c.representation_size = None

    c.eta = 0.2
    c.p = 4
    return c


config = get_b16_config()
vit_pretrain = 'ViT-B_16.npz'  # pretrained weights for backbone, please edit it for your configuration
epochs = 60
lr = 3e-4

zoom_size = 512
input_size = 448
batch_size = 10
which_set = 'cub'
if which_set == 'cub':
    train_csv = 'datasets/CUB/train.csv'
    test_csv = 'datasets/CUB/test.csv'
    train_root = 'datasets/CUB/images'
    test_root = 'datasets/CUB/images'
    config.num_classes = 200
elif which_set == 'dog':
    train_csv = 'datasets/StanfordDogs/train.csv'
    test_csv = 'datasets/StanfordDogs/test.csv'
    train_root = 'datasets/StanfordDogs/Images'
    test_root = 'datasets/StanfordDogs/Images'
    config.num_classes = 120
elif which_set == 'nabirds':
    train_csv = 'datasets/NABirds/train.csv'
    test_csv = 'datasets/NABirds/test.csv'
    train_root = 'datasets/NABirds/images'
    test_root = 'datasets/NABirds/images'
    config.num_classes = 555
elif which_set == 'iNat17':
    train_csv = 'datasets/iNat17/train.csv'
    test_csv = 'datasets/iNat17/test.csv'
    train_root = 'datasets/iNat17/'
    test_root = 'datasets/iNat17/'
    config.num_classes = 555
else:
    assert False, 'no dataset'

if which_set == 'dog':
    lr_ml = 100
    alpha = 0.01
else:
    lr_ml = 1
    alpha = 1

beta = [1, 1, 1, 1]
if which_set == 'cub':
    beta = [0.2, 0.2, 0.8, 0.8]


CUDA_VISIBLE_DEVICES = '0,1,2,3'
momentum = 0.9
weight_decay = 5e-4
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

weight_path = 'checkpoints/cub-20.pth'  # weight for val.py
