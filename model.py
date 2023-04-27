import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from os.path import join as pjoin
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """
    Possibly convert HWIO to OIHW
    :param weights: weights
    :param conv: transpose or not
    :return:
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Mlp(nn.Module):

    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
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

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        self.hybrid = None
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


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
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

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
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

        self.norm10 = LayerNorm(config.hidden_size, eps=1e-6)
        self.norm11 = LayerNorm(config.hidden_size, eps=1e-6)
        self.norm12 = LayerNorm(config.hidden_size, eps=1e-6)

        self.aof = Block(config)
        self.norm13 = LayerNorm(config.hidden_size, eps=1e-6)
        self.eta = config.eta

    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        weight = torch.eye(785).repeat(batch_size, 1, 1).cuda()
        eye = torch.eye(785).unsqueeze(0).cuda()

        out10 = None
        out11 = None

        layer_num = len(self.layer)
        for i in range(layer_num):
            hidden_states, w = self.layer[i](hidden_states)  # hidden_states: (batch, N+1, D)
            if i == 9:
                out10 = hidden_states
            elif i == 10:
                out11 = hidden_states
            w = w.detach().clone()
            w = w.mean(1)  # [batch_size, N+1, N+1]
            w = (w + 1.0 * eye) / 2
            w = w / w.sum(dim=-1, keepdim=True)
            weight = torch.matmul(w, weight)

        num = hidden_states.size(1) - 1  # Number of content tokens
        attn = weight[:, 0, 1:]
        _, idx = attn.sort(dim=-1, descending=True)
        idx = idx[:, :int(num * self.eta)] + 1  # The first token must be the class token
        cls_token = out11[:, 0].unsqueeze(1)  # [batch, 1, D]
        content_token = []
        for i in range(batch_size):
            content_token.append(out11[i][idx[i, :], :])  # (batch, [num*eta], D)
        content_token = torch.stack(content_token, dim=0)

        out = torch.cat([cls_token, content_token], dim=1)
        out_object, _ = self.aof(out.detach())
        out13 = self.norm13(out_object)

        out12 = self.norm12(hidden_states)

        out10 = self.norm10(out10)
        out11 = self.norm11(out11)
        return out10, out11, out12, out13, weight


class Transformer(nn.Module):

    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        return self.encoder(embedding_output)


class VisionTransformer(nn.Module):

    def __init__(self, config, img_size=224, num_classes=200):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier  # token
        self.transformer = Transformer(config, img_size)

        self.part_head10 = Linear(config.hidden_size, num_classes)
        self.part_head11 = Linear(config.hidden_size, num_classes)
        self.part_head12 = Linear(config.hidden_size, num_classes)
        self.part_head13 = Linear(config.hidden_size, num_classes)

        self.p = config.p
        self.img_size = img_size
        self.patch_size = config['patches']['size'][0]

    def forward(self, x, swap=False):
        out10, out11, out12, out13, weight = self.transformer(x)

        out10 = self.part_head10(out10[:, 0])
        out11 = self.part_head11(out11[:, 0])
        out12 = self.part_head12(out12[:, 0])
        out13 = self.part_head13(out13[:, 0])

        if swap:
            batch_size = x.size()[0]
            image = x

            p = self.img_size // self.patch_size
            image_new = image.detach().clone()
            weight = weight[:, 0, 1:]  # [batch, N]
            weight = weight / torch.amax(weight, dim=-1, keepdim=True)  # [batch, N]
            weight = weight.view(batch_size, 1, p, p)

            p = self.p
            ratio = self.img_size // p
            weight = F.adaptive_avg_pool2d(weight, (p, p)).view(batch_size, -1)  # [batch, p*p]
            weight = weight / weight.sum(dim=-1, keepdim=True)
            score, indices = torch.sort(weight, descending=True, dim=-1)  # [batch, N]
            rand_index = torch.randperm(batch_size).cuda()
            end = p ** 2

            lam_a = torch.ones(batch_size).cuda()
            lam_b = 1 - lam_a
            start = end // 2  # low attention
            for b in range(batch_size):
                for t in range(start, end):
                    idx = indices[b][t]
                    i, j = idx // p, idx % p
                    h1_start = i * ratio
                    h1_end = h1_start + ratio
                    w1_start = j * ratio
                    w1_end = w1_start + ratio

                    another_t = t - start
                    idx = indices[rand_index[b]][another_t]
                    i, j = idx // p, idx % p
                    h2_start = i * ratio
                    h2_end = h2_start + ratio
                    w2_start = j * ratio
                    w2_end = w2_start + ratio
                    image_new[b, :, h1_start:h1_end, w1_start:w1_end] = image[rand_index[b], :, h2_start:h2_end,
                                                                        w2_start: w2_end]
                    lam_a[b] -= score[b][t]
                    lam_b[b] += score[rand_index[b]][another_t]

            return out10, out11, out12, out13, image_new, lam_a, lam_b, rand_index
        else:
            return out10, out11, out12, out13

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":  # True
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
                if bname.startswith('layer'):
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)
                if bname.startswith('aof'):
                    block.load_from(weights, n_block=11)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
