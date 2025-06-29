#!/usr/bin/env python3
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials (pytorch-tutorial-master.zip)
# Adapted to multi/hyperspectral images: F. Arguello
# Readapted to FastViT: H. Carreira
# CNN21: 2 capas convolucionales, 1 completamente conectada
# oitaven WP (15%, texturas+fv+3kelm, t=3m44s): OA=93.03, OA=87.18
# CNN21 SEG EXP: 5 EPOCHS: 100 SAMPLES: [0.15, 0.05] ADA: 3 AUM: 1
# Class 01: 96.66+0.33
# Class 02: 80.72+2.00
# Class 03: 75.76+4.18
# Class 04: 87.01+3.79
# Class 05: 86.18+1.77
# Class 06: 92.11+1.21
# Class 07: 96.46+0.15
# Class 08: 95.77+0.21
# Class 09: 98.66+0.56
# Class 10: 90.81+0.62
# OA=94.77+0.20, AA=90.01+0.66, t=60 s

import math, random, struct, signal, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms
import argparse
import sys
from functools import partial
# from timm.models import register_model, model_entrypoint
from functools import partial
from modelos.fastvit_modificado import *
import ast

from modelos.models.modules.mobileone import reparameterize_model
from modelos.modelos_transformers.cait import CaiT
from modelos.models_crossformer.crossformer import CrossFormer
from modelos.models_swin.swin import SwinTransformer
from modelos.models_twins.twins import Twins
from modelos.models_convxnet.convxnet import ConvNeXt
from modelos.models_coanet.CoAtNetReducida import CoAtNet
from modelos.models_resnet.resnet import ResNet

# Los argumentos permiten que se pasen como parametros al script
parser = argparse.ArgumentParser(description='CNN21 FastViT Training')

parser.add_argument('--dataset', type=str, default='./ferreiras_river.raw', help='--dataset: Path o arquivo do dataset')

parser.add_argument('--gt', type=str, default='./ferreiras_river.pgm', help='--gt: Path o arquivo de GT (ground truth')

parser.add_argument('--seg', type=str, default='./seg_ferreiras_wp.raw', help='--seg: Path o arquivo de segmentacion')

parser.add_argument('--center', type=str, default='./seg_ferreiras_wp_centers.raw',
                    help='--center: Path o arquivo de centros da segmentacion')

# crear otro argumento que indique el modelo que deseamos usar: fastvit_t8, fastvit_t16, fastvit_s12, fastvit_sa12,fastvit_sa24,fastvit_sa36,fastvit_ma36
parser.add_argument('--model', type=str, default='fastvit_t8',
                    help='--model: Modelo FastViT a utilizar: fastvit_t8, fastvit_t16, fastvit_s12, fastvit_sa12,fastvit_sa24,fastvit_sa36,fastvit_ma36')

# Hacer que las los archivos de salida se posicionen en un directorio especico pasado como argumento
parser.add_argument('--output', type=str, default='./',
                    help='--output: Directorio donde se guardaran los archivos de salida')

# Bath size default 100
parser.add_argument('--batch', type=int, default=100, help='--batch: Tamaño del batc default=100')

# Learining rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
parser.add_argument('--ada', type=int, default=3,
                    help='--ada: Learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR')

# Valor de lr, default 0.001
parser.add_argument('--lr', type=float, default=0.0001, help='--lr: default 0.001')

# Metodo empregado para optimizar os pesos da rede na funcion de perda (loss)
parser.add_argument('--loss', type=str, default='cross_entropy',
                    help='Loss function to use: cross_entropy, mse, bce, bce_with_logits, l1, smooth_l1')

# Optimizador a utilizar: adam, sgd, rmsprop, adamw, adagrad, adadelta, adamax, asgd, lbfgs no axuste da rede
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer to use: adam, sgd, rmsprop, adamw, adagrad, adadelta, adamax, asgd, lbfgs')

parser.add_argument('--layers', type=str, default='2, 2, 4, 2',
                    help='--layers: Capas de la red a utilizar: [2, 2, 4, 2] ; [4, 4, 2, 2] ; [2, 2, 2, 2]')
parser.add_argument('--embed_dims', type=str, default='48, 96, 192, 384',
                    help='--embed_dims: Dimensiones de la red a utilizar: [48, 96, 192, 384] ; [384, 192, 96, 48]')
parser.add_argument('--mlp_ratios', type=str, default='3, 3, 3, 3',
                    help='--mlp_ratios: Ratios de la red a utilizar: [3, 3, 3, 3] ; [2, 2, 2, 2]')
parser.add_argument('--token_mixers', type=str, default='("repmixer", "repmixer", "repmixer", "repmixer")',
                    help='--token_mixers: Mezcladores de la red a utilizar: ("repmixer", "repmixer", "repmixer", "repmixer") ; ("repmixer", "repmixer", "repmixer", "attention"); ("attention", "repmixer", "repmixer", "repmixer")')

parser.add_argument('--group-size', type=str, default='4, 2, 1',
                    help='--group-size: Tamaño de grupo para CrossFormer: [4, 2, 1]')
parser.add_argument('--crs-interval', type=str, default='2, 1, 1',
                    help='--crs-interval: Intervalo de CRS para CrossFormer: [2, 1, 1]')
parser.add_argument('--embed-dim-inicial', type=int, default=96,
                    help='--embed-dim: Dimensión de incrustación para CrossFormer: 96')
parser.add_argument('--mpl-ratio', type=float, default=4., help='--mpl-ratio: Tasa de MLP para CrossFormer: 4')
parser.add_argument('--num-heads', type=str, default='3, 6, 12',
                    help='--num-heads: Número de cabezas para CrossFormer: [3, 6, 12]')
parser.add_argument('--patch-size', type=str, default='[2]',
                    help='--patch-size: Tamaño de parche para CrossFormer: [2]')

parser.add_argument('--num-blocks', type=str, default='2, 2, 3, 5, 2',
                    help='--num-blocks: Bloques de CoAtNet: [2, 2, 3, 5, 2]')
parser.add_argument('--block-types', type=str, default='C T C T',
                    help='--block-types: Tipos de bloques de CoAtNet: ["C", "T", "C", "T"]')
parser.add_argument('--sr-ratios', type=str, default='', help='    ')
parser.add_argument('--window-size', type=int, default=2,
                    help='--window-size: Tamaño de ventana para SwinTransformer: 2')

args = parser.parse_args()


def parse_layers(layers_str):
    try:
        return list(map(int, ast.literal_eval(layers_str)))
    except (ValueError, SyntaxError):
        return list(map(int, layers_str.split(',')))


def ensure_list(val):
    if isinstance(val, list):
        return val
    try:
        v = ast.literal_eval(val)
        if isinstance(v, list):
            return v
        else:
            return [v]
    except Exception:
        return [val]


layers = parse_layers(args.layers)
mlp_ratios = ensure_list(args.mlp_ratios)

if (args.model == 'twins_universal'):
    embed_dims = ensure_list(args.embed_dims)[0]
    layers = ensure_list(args.layers)[0]
    sr_ratios = ensure_list(args.sr_ratios)[0]
    patch_size = ast.literal_eval(args.patch_size)
else:
    token_mixers = ensure_list(args.token_mixers)
    group_size = parse_layers(args.group_size)
    crs_interval = parse_layers(args.crs_interval)
    num_blocks = ensure_list(args.num_blocks)
    embed_dims = parse_layers(args.embed_dims)
    num_heads = parse_layers(args.num_heads)
    patch_size = ensure_list(args.patch_size)
    num_heads = parse_layers(args.num_heads)


def parse_block_types(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return s.split()


block_types = parse_block_types(args.block_types)


# Definición CoAtNet universal
def coatnet_universal(pretrained=False, **kwargs):
    num_blocks = layers
    channels = embed_dims
    print(f"num_blocks: {num_blocks}, channels: {channels}, block_types: {block_types}")
    model = CoAtNet(
        (32, 32), 5, num_blocks, channels, num_classes=10, block_types=block_types
    )
    return model


# Definición CrossFormer universal
def crossformer_universal(pretrained=False, **kwargs):
    print(
        f"patch_size: {patch_size}, in_chans: 5, num_classes: 10, embed_dim: {args.embed_dim_inicial}, depths: {layers}, num_heads: {num_heads}, group_size: {group_size}, crs_interval: {crs_interval}, mlp_ratio: {args.mpl_ratio}"),
    model = CrossFormer(
        img_size=32, patch_size=patch_size, in_chans=5, num_classes=10,
        embed_dim=args.embed_dim_inicial, depths=layers, num_heads=num_heads,
        group_size=group_size, crs_interval=crs_interval, mlp_ratio=args.mpl_ratio, merge_size=[[2], [2]]
    )
    return model

# Definicion de Swin Universal
def swin_universal(pretrained=False, **kwargs):
    model = SwinTransformer(img_size=32,
                            patch_size=2,
                            in_chans=5,
                            num_classes=10,
                            embed_dim=args.embed_dim_inicial,
                            depths=layers,
                            num_heads=num_heads,
                            window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                            use_checkpoint=False, fused_window_process=False)
    return model

# Definición de twins universal
def twins_universal(pretrained=False, **kwargs):
    print("patch_size, embed_dims, mlp_ratios, layers, sr_ratios")
    print(patch_size, embed_dims, mlp_ratios, layers, sr_ratios)
    # Definición de los parámetros del modelo Twins
    model = Twins(
        img_size=32,
        patch_size=patch_size,
        in_chans=5,
        num_classes=10,
        global_pool='avg',
        embed_dims=embed_dims,
        mlp_ratios=[4] * len(layers),
        depths=layers,
        sr_ratios=sr_ratios,
    )
    return model

# Contar parámetros FastViT
def fastvit_universal(pretrained=False, **kwargs):
    downsamples = [True] * len(layers)
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    if pretrained:
        raise ValueError("Funcionalidade non implementada... sentímolo :(")
    return model


# enviar un aviso si no se pasan los argumentos
if len(sys.argv) < 5:
    # mostrar un mensaje de ayuda si no se pasan argumentos
    print('Debes enviar como argumento al menos el dataset, el GT, la segmentación y los centros de la segmentación')
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

EXP = 3  # numero de experimentos
EPOCHS = 100  # EPOCHS de entrenamiente del clasificador, default=100
SAMPLES = [0.15, 0.05]  # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05)
BATCH = args.batch  # batch_size, defecto 100
ADA = args.ada  # learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
AUM = 1  # aumentado: 0-sin_aumentado, 1-con_aumentado
DET = 0  # experimentos: 0-aleatorios, 1-deterministas
TEST = 1  # 0-validacion, 1-test

origi_min = 0
origi_max = 0

DATASET = args.dataset
GT = args.gt
SEG = args.seg
CENTER = args.center

model_registry = {}

reference_time = None
mean_time = 0


def register_model(name, model_fn):
    global model_registry
    if name in model_registry:
        raise ValueError(f"O modelo ({name}) xa se atopa rexistrado. Volve a rexistrarse igual.")
    model_registry[name] = model_fn


def is_model_registered(model_name):
    global model_registry
    return model_name in model_registry

register_model('twins_universal', twins_universal)
register_model('swin_universal', swin_universal)
register_model('crossformer_universal', crossformer_universal)
register_model('coatnet_universal', coatnet_universal)
register_model('fastvit_universal', fastvit_universal)

# -----------------------------------------------------------------
# PYTORCH - NETWORK
# -----------------------------------------------------------------

import timm
import torch.nn as nn

# -----------------------------------------------------------------
# PYTORCH - MAIN
# -----------------------------------------------------------------

import timm
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
import argparse


# -----------------------------------------------------------------
# PYTORCH - MAIN
# -----------------------------------------------------------------


def obtener_modelo(model_name, pretrained=False, **kwargs):
    global model_registry
    if model_name in model_registry:
        return model_registry[model_name](pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is not registered in the registry.")


def contarParametros():
    model = obtener_modelo(args.model, pretrained=False, num_classes=10)

    # Número total de parámetros
    total_params = sum(p.numel() for p in model.parameters())

    # Número de parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total de parámetros: {total_params}")
    print(f"Parámetros entrenables: {trainable_params}")


if __name__ == '__main__':
    contarParametros()
