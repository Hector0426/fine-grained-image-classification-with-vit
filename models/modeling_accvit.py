# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        
        self.token_filter = config.token_filter
        if self.token_filter:
            layer = Block(config)
            self.token_filter_module = copy.deepcopy(layer)
            self.token_filter_module_norm = LayerNorm(config.hidden_size, eps=1e-6)
        else:
            self.token_filter_module = nn.Identity()
            self.token_filter_module_norm = nn.Identity()

        self.eta = config['token_filter_eta']
        self.out_indices = config['out_indices']

        for index in range(len(self.out_indices)):
            setattr(self, f'encoder_norm_{self.out_indices[index]}', LayerNorm(config.hidden_size, eps=1e-6))

    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        # print(hidden_states.shape)
        token_number = hidden_states.shape[1]
        weight = torch.eye(token_number).repeat(batch_size, 1, 1).cuda()
        eye = torch.eye(token_number).unsqueeze(0).cuda()

        layer_num = len(self.layer)
        outs = []
        out_filter_input = None
        for i in range(layer_num):
            hidden_states, w = self.layer[i](hidden_states)  # hidden_states: (batch, N+1, D)
            if i in self.out_indices:
                if i == 11:
                    outs.append(self.encoder_norm(hidden_states)[:, 0])  # cls token
                else:
                    outs.append(
                        getattr(self, f'encoder_norm_{i}')(hidden_states)[:, 0]
                    ) # cls token
            if i == layer_num - 2:
                out_filter_input = hidden_states
            w = w.detach().clone()
            w = w.mean(1)  # [batch_size, N+1, N+1]
            w = (w + 1.0 * eye) / 2
            w = w / w.sum(dim=-1, keepdim=True)
            weight = torch.matmul(w, weight)

        if self.token_filter:
            num = out_filter_input.size(1) - 1  # Number of content tokens
            attn = weight[:, 0, 1:]
            _, idx = attn.sort(dim=-1, descending=True)
            idx = idx[:, :int(num * self.eta)] + 1  # The first token must be the class token
            cls_token = out_filter_input[:, 0].unsqueeze(1)  # [batch, 1, D]
            content_token = []
            for i in range(batch_size):
                content_token.append(out_filter_input[i][idx[i, :], :])  # (batch, [num*eta], D)
            content_token = torch.stack(content_token, dim=0)
            out = torch.cat([cls_token, content_token], dim=1)
            out_filter, _ = self.token_filter_module(out)
            outs.append(self.token_filter_module_norm(out_filter)[:, 0])

        return outs, weight

    
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        return self.encoder(embedding_output)


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        # self.head = Linear(config.hidden_size, num_classes)

        self.out_indices = config.out_indices

        for index in config.out_indices:
            setattr(self, f'head_{index}', Linear(config.hidden_size, num_classes))
        if config.token_filter:
            self.head_filter = nn.Sequential(
                LayerNorm(config.hidden_size, eps=1e-6),
                Linear(config.hidden_size, num_classes)
            )
        else:
            self.head_filter = nn.Identity()

        self.token_filter = config.token_filter
        self.block_size = config.block_size
        self.img_size = (448, 448)
        self.patch_size = (16, 16)

        self.swap = config.swap
        self.second_area = config.second_area
        self.generate_label = config.generate_label
        self.pad = config.pad

    def forward(self, x, labels=None, stage='first'):
        outs, weight = self.transformer(x)

        out_list = []
        for i in range(len(self.out_indices)):
            if self.out_indices[i] == 11:
                out = getattr(self, f'head_{self.out_indices[i]}')(outs[i])
            else:
                out = getattr(self, f'head_{self.out_indices[i]}')(outs[i].detach())
            out_list.append(out)

        if self.token_filter:
            assert len(outs) == len(self.out_indices) + 1
            out = self.head_filter(outs[-1])
            out_list.append(out)

        if not self.training:
            return out_list
        

        if self.swap and stage == 'first':
            batch_size = x.size()[0]
            image = x

            patch_num = self.img_size[0] // self.patch_size[0]
            image_new = image.detach().clone()
            weight = weight[:, 0, 1:]  # [batch, N]
            weight = weight / torch.amax(weight, dim=-1, keepdim=True)  # [batch, N]
            weight = weight.view(batch_size, 1, patch_num, patch_num)

            ratio = self.img_size[0] // self.block_size
            weight = F.adaptive_avg_pool2d(weight, (self.block_size, self.block_size)).view(batch_size, -1)
            weight = weight / weight.sum(dim=-1, keepdim=True)
            score, indices = torch.sort(weight, descending=True, dim=-1)  # [batch, N]


            block_num = self.block_size ** 2  
            rand_index = torch.randperm(batch_size).cuda() 
            lam_a = torch.ones(batch_size).cuda()  
            lam_b = 1 - lam_a
            start = round(block_num * (1 - self.second_area)) 
            swap_number = block_num - start  
            block_area_weight = 1 / block_num  

            for b in range(batch_size):
                if b == rand_index[b]:
                    continue

                if self.pad == 'order':
                    swap_rand_index = torch.arange(swap_number, dtype=torch.uint8).cuda()
                else:
                    swap_rand_index = torch.randperm(swap_number).cuda()

                for t in range(start, block_num):
                    idx = indices[b][t]
                    i, j = idx // self.block_size, idx % self.block_size
                    h1_start = i * ratio
                    h1_end = h1_start + ratio
                    w1_start = j * ratio
                    w1_end = w1_start + ratio

                    another_t = swap_rand_index[t - start].item()
                    idx = indices[rand_index[b].item()][another_t]
                    i, j = idx // self.block_size, idx % self.block_size
                    h2_start = i * ratio
                    h2_end = h2_start + ratio
                    w2_start = j * ratio
                    w2_end = w2_start + ratio
                    image_new[b, :, h1_start:h1_end, w1_start:w1_end] = \
                        image[rand_index[b], :, h2_start:h2_end, w2_start: w2_end]

                    if self.generate_label == 'weight':
                        lam_a[b] -= score[b][t]
                        lam_b[b] += score[rand_index[b]][another_t]
                    else:
                        lam_a[b] -= block_area_weight
                        lam_b[b] += block_area_weight

            return out_list, image_new, lam_a, lam_b, rand_index
        else:
            return out_list

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                # if bname.startswith('part') == False:
                #     for uname, unit in block.named_children():
                #         unit.load_from(weights, n_block=uname)
        
                for uname, unit in block.named_children():
                    if isinstance(unit, Block):
                        unit.load_from(weights, n_block=uname)
                if bname.startswith('token_filter_module'):
                    if isinstance(block, Block):
                        block.load_from(weights, n_block=11)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) 

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

CONFIGS = {
    # 'ViT-B_16': configs.get_b16_config(),
    'ViT-B_16': configs.get_b16_accvit_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
