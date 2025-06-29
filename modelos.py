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
from torch.utils.data import Dataset,DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms
import argparse
import sys
from functools import partial
#from timm.models import register_model, model_entrypoint
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

parser.add_argument('--center', type=str, default='./seg_ferreiras_wp_centers.raw', help='--center: Path o arquivo de centros da segmentacion')

# crear otro argumento que indique el modelo que deseamos usar: fastvit_t8, fastvit_t16, fastvit_s12, fastvit_sa12,fastvit_sa24,fastvit_sa36,fastvit_ma36
parser.add_argument('--model', type=str, default='fastvit_t8', help='--model: Modelo FastViT a utilizar: fastvit_t8, fastvit_t16, fastvit_s12, fastvit_sa12,fastvit_sa24,fastvit_sa36,fastvit_ma36')

# Hacer que las los archivos de salida se posicionen en un directorio especico pasado como argumento
parser.add_argument('--output', type=str, default='./', help='--output: Directorio donde se guardaran los archivos de salida')

# Bath size default 100
parser.add_argument('--batch', type=int, default=100, help='--batch: Tamaño del batc default=100')

# Learining rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
parser.add_argument('--ada', type=int, default=3, help='--ada: Learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR')

# Valor de lr, default 0.001
parser.add_argument('--lr', type=float, default=0.0001, help='--lr: default 0.001')

# Metodo empregado para optimizar os pesos da rede na funcion de perda (loss)
parser.add_argument('--loss', type=str, default='cross_entropy', help='Loss function to use: cross_entropy, mse, bce, bce_with_logits, l1, smooth_l1')

# Optimizador a utilizar: adam, sgd, rmsprop, adamw, adagrad, adadelta, adamax, asgd, lbfgs no axuste da rede
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use: adam, sgd, rmsprop, adamw, adagrad, adadelta, adamax, asgd, lbfgs')

# FastViT
parser.add_argument('--layers', type=str, default='2, 2, 4, 2', help='--layers: Capas de la red a utilizar: [2, 2, 4, 2] ; [4, 4, 2, 2] ; [2, 2, 2, 2]')
parser.add_argument('--embed_dims', type=str, default='48, 96, 192, 384', help='--embed_dims: Dimensiones de la red a utilizar: [48, 96, 192, 384] ; [384, 192, 96, 48]')
parser.add_argument('--mlp_ratios', type=str, default='3, 3, 3, 3', help='--mlp_ratios: Ratios de la red a utilizar: [3, 3, 3, 3] ; [2, 2, 2, 2]')
parser.add_argument('--token_mixers', type=str, default='("repmixer", "repmixer", "repmixer", "repmixer")', help='--token_mixers: Mezcladores de la red a utilizar: ("repmixer", "repmixer", "repmixer", "repmixer") ; ("repmixer", "repmixer", "repmixer", "attention"); ("attention", "repmixer", "repmixer", "repmixer")')

parser.add_argument('--group-size', type=str, default='4, 2, 1', help='--group-size: Tamaño de grupo para CrossFormer: [4, 2, 1]')
parser.add_argument('--crs-interval', type=str, default='2, 1, 1', help='--crs-interval: Intervalo de CRS para CrossFormer: [2, 1, 1]')
parser.add_argument('--embed-dim-inicial', type=int, default=96, help='--embed-dim: Dimensión de incrustación para CrossFormer: 96')
parser.add_argument('--mpl-ratio', type=float, default=4., help='--mpl-ratio: Tasa de MLP para CrossFormer: 4')
parser.add_argument('--num-heads', type=str, default='3, 6, 12', help='--num-heads: Número de cabezas para CrossFormer: [3, 6, 12]')
parser.add_argument('--patch-size', type=str, default='[2]', help='--patch-size: Tamaño de parche para CrossFormer: [2]')


# CoAtNet
parser.add_argument('--num-blocks', type=str, default='2, 2, 3, 5, 2', help='--num-blocks: Bloques de CoAtNet: [2, 2, 3, 5, 2]')
parser.add_argument('--block-types', type=str, default='C T C T', help='--block-types: Tipos de bloques de CoAtNet: ["C", "T", "C", "T"]')
parser.add_argument('--sr-ratios', type=str, default='2, 2, 2', help='--sr-ratios: Ratios de SR para CoAtNet: [2, 2, 2]')

parser.add_argument('--window-size', type=int, default=2, help='--window-size: Tamaño de ventana para SwinTransformer: 2')

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


def parse_block_types(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return s.split()

block_types = parse_block_types(args.block_types)

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
    print(f"patch_size: {patch_size}, in_chans: 5, num_classes: 10, embed_dim: {args.embed_dim_inicial}, depths: {layers}, num_heads: {num_heads}, group_size: {group_size}, crs_interval: {crs_interval}, mlp_ratio: {args.mpl_ratio}"),
    model = CrossFormer(
       img_size=32,patch_size=patch_size, in_chans=5, num_classes=10,
       embed_dim=args.embed_dim_inicial, depths=layers, num_heads=num_heads,
       group_size=group_size , crs_interval=crs_interval, mlp_ratio=args.mpl_ratio, merge_size=[[2], [2]]
    )
    return model

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


# enviar un aviso si no se pasan los argumentos
if len(sys.argv) < 5:
    # mostrar un mensaje de ayuda si no se pasan argumentos
    print('Debes enviar como argumento al menos el dataset, el GT, la segmentación y los centros de la segmentación')
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

EXP=3      # numero de experimentos
EPOCHS=100 # EPOCHS de entrenamiente del clasificador, default=100
SAMPLES=[0.15,0.05] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05) 
BATCH=args.batch  # batch_size, defecto 100
ADA=args.ada  # learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
AUM=1  # aumentado: 0-sin_aumentado, 1-con_aumentado
DET=0  # experimentos: 0-aleatorios, 1-deterministas
TEST=1 # 0-validacion, 1-test

origi_min = 0
origi_max = 0

DATASET = args.dataset
GT = args.gt
SEG = args.seg
CENTER = args.center

model_registry = {}

reference_time = None
mean_time = 0

def register_model(name,model_fn):
  global model_registry
  if name in model_registry:
    raise ValueError(f"O modelo ({name}) xa se atopa rexistrado. Volve a rexistrarse igual.")
  model_registry[name] = model_fn

def is_model_registered(model_name):
  global model_registry
  return model_name in model_registry

def fastvit_t8(pretrained=False, **kwargs):
    layers = [2, 2, 4, 2]
    embed_dims = [48, 96, 192, 384]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_t"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


def fastvit_t12(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_t"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def fastvit_s12(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def fastvit_sa12(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def fastvit_sa24(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def fastvit_sa36(pretrained=False, **kwargs):
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def fastvit_ma36(pretrained=False, **kwargs):
    layers = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    # model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def cait(pretrained=False, **kwargs):
  model = CaiT(
      image_size = 32,
      patch_size = 4,
      dim = 96,            # 96-128 para imaxes pequenas
      depth = 6,           # 6-8 capas para parches
      cls_depth = 2,       # 2-4 capas para atención de clase
      heads = 8,           # 4-8 cabezas
      mlp_dim = 384,       # 4*dim = 4*96 = 384
      dropout = 0.1,
      emb_dropout = 0.1,
      **kwargs             # Eliminar layer_dropout
  )
  return model

def crossformer(pretrained=False, **kwargs):
    model = CrossFormer(**kwargs)
    return model

def crossformer_15M_CF_01(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[2, 3, 3], num_heads=[3, 6, 12],
       group_size=[4, 2, 1], crs_interval=[2, 1, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_02(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=48, depths=[2, 3, 3], num_heads=[3, 6, 12],
       group_size=[4, 2, 1], crs_interval=[2, 1, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_03(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=48, depths=[2, 2, 2], num_heads=[3, 6, 12],
       group_size=[4, 2, 1], crs_interval=[2, 1, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_04(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[2, 2, 2], num_heads=[2, 4, 8],
       group_size=[4, 2, 1], crs_interval=[2, 1, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_05(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[4], in_chans=5, num_classes=10,
       embed_dim=96, depths=[2, 2, 2], num_heads=[2, 4, 8],
       group_size=[4, 2, 1], crs_interval=[2, 1, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_06(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[4], in_chans=5, num_classes=10,
       embed_dim=96, depths=[2, 3, 3], num_heads=[2, 4, 8],
       group_size=[4, 2, 1], crs_interval=[2, 1, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_07(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[2, 2, 2], num_heads=[2, 4, 8],
       group_size=[4, 2, 1], crs_interval=[4, 2, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_08(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=48, depths=[2, 2, 2], num_heads=[2, 4, 8],
       group_size=[4, 2, 1], crs_interval=[4, 2, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_09(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[4, 4, 2], num_heads=[2, 4, 8],
       group_size=[4, 2, 1], crs_interval=[4, 2, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_10(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[4, 4, 2], num_heads=[3, 6, 12],
       group_size=[4, 2, 1], crs_interval=[4, 2, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_11(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[4, 3, 2], num_heads=[3, 6, 12],
       group_size=[4, 2, 1], crs_interval=[4, 2, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def crossformer_15M_CF_12(pretrained=False, **kwargs):
    model = CrossFormer(
       patch_size=[2], in_chans=5, num_classes=10,
       embed_dim=96, depths=[3, 4, 2], num_heads=[3, 6, 12],
       group_size=[4, 2, 1], crs_interval=[4, 2, 1], mlp_ratio=4.,merge_size=[[2], [2]]
    )
    return model

def swin(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[2, 2, 6], 
                          num_heads=[3, 6, 12],
                          window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_1(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[2, 2], 
                          num_heads=[3, 6],
                          window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_2(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=4, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[2, 2], 
                          num_heads=[3, 6],
                          window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_3(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[2, 2], 
                          num_heads=[3, 6],
                          window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_4(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[3, 6], 
                          num_heads=[3, 6],
                          window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_5(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[2, 2], 
                          num_heads=[3, 6],
                          window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_6(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=96, 
                          depths=[2, 2], 
                          num_heads=[3, 6],
                          window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False
  )
  return model

def swin_4M_SWT_7(pretrained=False, **kwargs):
  model = SwinTransformer(img_size=32, 
                          patch_size=2, 
                          in_chans=5, 
                          num_classes=10,
                          embed_dim=48, 
                          depths=[1, 1], 
                          num_heads=[3, 6],
                          window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, fused_window_process=False)
  return model


def twins(pretrained=False, **kwargs):
  model = Twins(
     img_size=32,
      patch_size=4,
      in_chans=5,
      num_classes=10,
      global_pool='avg',
      embed_dims=(64, 128, 256, 512),
      num_heads=(2, 2, 4, 8),
      mlp_ratios=(4, 4, 4, 4),
      depths=(3, 4, 6, 3),
      sr_ratios=(2, 2, 1, 1),
      wss=None,
      drop_rate=0.,
      pos_drop_rate=0.,
      proj_drop_rate=0.,
      attn_drop_rate=0.,
      drop_path_rate=0.,
      norm_layer=partial(nn.LayerNorm, eps=1e-6)
  )
  return model

def convxnet(pretrained=False, **kwargs):
    # def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
    model = ConvNeXt(in_chans=5, num_classes=10, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6, 
                 head_init_scale=1.
    )                     # Tamaño del parche
    return model

def coatnet(pretrained=False, **kwargs):
    num_blocks = [2, 2, 3, 5, 2]      
    channels = [64, 96, 192, 384, 768]       
    model = CoAtNet(
       (32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C', 'T']
    )
    return model

def coatnet_a01(pretrained=False, **kwargs):
    num_blocks = [2, 2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C', 'T'])
    return model

def coatnet_a02(pretrained=False, **kwargs):
    num_blocks = [1, 1, 2, 2, 1]
    channels = [64, 96, 192, 384, 768]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C', 'T'])
    return model

def coatnet_a03(pretrained=False, **kwargs):
    num_blocks = [1, 1, 1, 1, 1]
    channels = [64, 96, 192, 384, 768]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C', 'T'])
    return model

def coatnet_a04(pretrained=False, **kwargs):
    num_blocks = [1, 1, 2, 2]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C'])
    return model

def coatnet_a05(pretrained=False, **kwargs):
    num_blocks = [1, 1, 1, 1]
    channels = [32, 64, 128, 256]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C'])
    return model

def coatnet_a06(pretrained=False, **kwargs):
    num_blocks = [2, 2, 2, 2]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C'])
    return model

def coatnet_a07(pretrained=False, **kwargs):
    num_blocks = [1, 2, 2, 1]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C'])
    return model

def coatnet_a08(pretrained=False, **kwargs):
    num_blocks = [1, 2, 2, 1]
    channels = [32, 48, 96, 192]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'T', 'C'])
    return model

def coatnet_a09(pretrained=False, **kwargs):
    num_blocks = [1, 1, 2, 2, 1]
    channels = [64, 96, 192, 384, 768]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'T', 'T', 'T'])
    return model

def coatnet_a10(pretrained=False, **kwargs):
    num_blocks = [1, 1, 2, 2, 1]
    channels = [64, 96, 192, 384, 768]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['C', 'C', 'C', 'C'])
    return model

def coatnet_a11(pretrained=False, **kwargs):
    num_blocks = [1, 2, 2, 1]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'T', 'C'])
    return model

def coatnet_a12(pretrained=False, **kwargs):
    num_blocks = [2, 2, 1, 1]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'T', 'C'])
    return model

def coatnet_a13(pretrained=False, **kwargs):
    num_blocks = [3, 2, 1, 1]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'T', 'C'])
    return model

def coatnet_a14(pretrained=False, **kwargs):
    num_blocks = [1, 1, 1, 1]
    channels = [64, 96, 192, 384]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'T', 'C'])
    return model

def coatnet_a15(pretrained=False, **kwargs):
    num_blocks = [1, 1, 1]
    channels = [64, 96, 192]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'C'])
    return model

def coatnet_a16(pretrained=False, **kwargs):
    num_blocks = [1, 1, 1]
    channels = [64, 96, 192]
    model = CoAtNet((32, 32), 5, num_blocks, channels, num_classes=10, block_types=['T', 'T'])
    return model

def resnet(pretrained=False, **kwargs):
    model = ResNet(
        layers=[2, 5, 5, 2],
        num_classes=10
    )
    return model

def CNN21(pretrained=False, **kwargs):
    # CNN21: 2 capas convolucionales, 1 completamente conectada
    model = cnn()
    return model

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
      mlp_ratios=[4]* len(layers),
      depths=layers,
      sr_ratios=sr_ratios,
  )
  return model

# Modelo base CNN21
class cnn(nn.Module):
  def __init__(self,N1=5,N2=16,N3=32,N5=10,D1=2,D2=2):
    super(cnn,self).__init__()
    # 5. Red: CNN con dos capas convolucionales y una lineal
    # 5.1. capa conv.1
    H1=32      # lado patches entrada, por defecto 28 (sizex=sizey)
    H2=int(H1/D1) # lado patches salida (calculada), por defecto 28 (sizex=sizey)

    # 5.2. capa conv.2, parametros de entrada N2,H2 vienen dados por la capa anterior
    D2=2          # decimacion, por defecto 2
    H3=int(H2/D2) # lado patches salida
      
    # 5.3. capa completamente conectada, parametro de entrada N4 viene de la etapa anterior 
    N4=H3*H3*N3   # dimension de entrada
    
    self.layer1=nn.Sequential(
      nn.Conv2d(N1,N2,kernel_size=3,stride=1,padding=2),
      nn.BatchNorm2d(N2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D1))
    self.layer2=nn.Sequential(
      nn.Conv2d(N2,N3,kernel_size=5,stride=1,padding=2),
      nn.BatchNorm2d(N3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D2))
    self.fc=nn.Linear(N4,N5)
      
  def forward(self,x):
    out=self.layer1(x)
    out=self.layer2(out)
    out=out.reshape(out.size(0),-1)
    out=self.fc(out)
    return out

# Registra los modelos
register_model('fastvit_t8', fastvit_t8)
register_model('fastvit_t12', fastvit_t12)
register_model('fastvit_s12', fastvit_s12)
register_model('fastvit_sa12', fastvit_sa12)
register_model('fastvit_sa24', fastvit_sa24)
register_model('fastvit_sa36', fastvit_sa36)
register_model('fastvit_ma36', fastvit_ma36)
register_model('cait', cait)
register_model('crossformer', crossformer)
register_model('twins', twins)
register_model('swin', swin)
register_model('convxnet', convxnet)
register_model('coatnet', coatnet)
register_model('coatnet_a01', coatnet_a01)
register_model('coatnet_a02', coatnet_a02)
register_model('coatnet_a03', coatnet_a03)
register_model('coatnet_a04', coatnet_a04)
register_model('coatnet_a05', coatnet_a05)
register_model('coatnet_a06', coatnet_a06)
register_model('coatnet_a07', coatnet_a07)
register_model('coatnet_a08', coatnet_a08)
register_model('coatnet_a09', coatnet_a09)
register_model('coatnet_a10', coatnet_a10)
register_model('coatnet_a11', coatnet_a11)
register_model('coatnet_a12', coatnet_a12)
register_model('coatnet_a13', coatnet_a13)
register_model('coatnet_a14', coatnet_a14)
register_model('coatnet_a15', coatnet_a15)
register_model('coatnet_a16', coatnet_a16)

# Modelos xenericos
register_model('coatnet_universal', coatnet_universal)
register_model('crossformer_universal', crossformer_universal)
register_model('twins_universal', twins_universal)
register_model('swin_universal', swin_universal)

# Registra los modelos específicos de CrossFormer y Swin
register_model('cnn', CNN21)
register_model('resnet', resnet)
register_model('swin_4M_SWT_1', swin_4M_SWT_1)
register_model('swin_4M_SWT_2', swin_4M_SWT_2)
register_model('swin_4M_SWT_3', swin_4M_SWT_3)
register_model('swin_4M_SWT_4', swin_4M_SWT_4)
register_model('swin_4M_SWT_5', swin_4M_SWT_5)
register_model('swin_4M_SWT_6', swin_4M_SWT_6)
register_model('swin_4M_SWT_7', swin_4M_SWT_7)
register_model('crossformer_15M_CF_01', crossformer_15M_CF_01)
register_model('crossformer_15M_CF_02', crossformer_15M_CF_02)
register_model('crossformer_15M_CF_03', crossformer_15M_CF_03)
register_model('crossformer_15M_CF_04', crossformer_15M_CF_04)
register_model('crossformer_15M_CF_05', crossformer_15M_CF_05)
register_model('crossformer_15M_CF_06', crossformer_15M_CF_06)
register_model('crossformer_15M_CF_07', crossformer_15M_CF_07)
register_model('crossformer_15M_CF_08', crossformer_15M_CF_08)
register_model('crossformer_15M_CF_09', crossformer_15M_CF_09)
register_model('crossformer_15M_CF_10', crossformer_15M_CF_10)
register_model('crossformer_15M_CF_11', crossformer_15M_CF_11)
register_model('crossformer_15M_CF_12', crossformer_15M_CF_12)



#-----------------------------------------------------------------
# FUNCIONES PARA LEER DATASETS Y SELECCIONAR MUESTRAS
#-----------------------------------------------------------------

def read_raw(fichero):
  global origi_min, origi_max
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  print('* Read dataset:',fichero)
  print('  B:',B,'H:',H,'V:',V)
  print('  Read:',len(datos))
  origi_min = datos.min()
  origi_max = datos.max()
  print('  Maximo df entrada:',origi_max,' Minimo df entrada:',origi_min)  # esta red no necesita realmente normalizar
  print("# ---------------- Normalizando datos ---------------- #")
  datos=preprocessing.minmax_scale(datos)
  print('  min:',datos.min(),'max:',datos.max())
  print("# -------------- Normalización finalizada ------------ #")
  datos=datos.reshape(V,H,B)
  datos=torch.FloatTensor(datos)
  return(datos,H,V,B)

def read_seg(fichero):
  (H,V)=np.fromfile(fichero,count=2,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=2*4,dtype=np.uint32)
  print('* Read segmentation:',fichero)
  print('  H:',H,'V:',V)
  print('  Read:',len(datos))
  return(datos,H,V)

def read_seg_centers(fichero):
  (H,V,nseg)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=3*4,dtype=np.uint32)
  print('* Read centers:',fichero)
  print('  H:',H,'V:',V,'nseg',nseg)
  print('  Read:',len(datos))
  return(datos,H,V,nseg)

def desnormalizarDatos(datos):
  # desnormalizamos los datos
  datos = datos * (origi_max - origi_min) + origi_min
  return datos

def save_raw_noverbose(output,H,V,B,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    f.write(struct.pack('i',B))
    f.write(struct.pack('i',H))
    f.write(struct.pack('i',V))
    output=output.reshape(H*V*B)
    for i in range(H*V*B):
      f.write(struct.pack('i',int(output[i])))
    f.close()

def save_raw(output,H,V,B,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    f.write(struct.pack('i',B))
    f.write(struct.pack('i',H))
    f.write(struct.pack('i',V))
    output=output.reshape(H*V*B)
    for i in range(H*V*B):
      f.write(struct.pack('i',np.int(output[i])))
    f.close()
    print('* Saved file:',filename)

def read_pgm(fichero):
  try:
    pgmf=open(fichero,"rb")
  except IOError:
    print('No puedo abrir ',fichero)
  else:
    assert pgmf.readline().decode()=='P5\n'
    line=pgmf.readline().decode()
    while(line[0]=='#'):
      line=pgmf.readline().decode()
    (H,V)=line.split()
    H=int(H); V=int(V)
    depth=int(pgmf.readline().decode())
    assert depth<=255
    raster=[]
    for i in range(H*V):
      raster.append(ord(pgmf.read(1)))
    print('* Read GT:',fichero)
    print('  H:',H,'V:',V,'depth:',depth)
    print('  Read:',len(raster))
    return(raster,H,V)

def save_pgm(output,H,V,nclases,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    # f.write(b'P5\n')
    cadena='P5\n'+str(H)+' '+str(V)+'\n'+str(nclases)+'\n'
    f.write(bytes(cadena,'utf-8'))
    f.write(output)
    f.close()
    print('* Saved file:',filename)

def select_patch(datos,sizex,sizey,x,y):
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  patch=datos[:,y1:y2,x1:x2]
  return(patch)

def seg_center(seg,H,V):
  print('* Segment centers (tarda mucho)')
  nseg=0
  for i in range(H*V):
    if(seg[i]>nseg): nseg=seg[i]
  nseg=nseg+1
  print('  segments:',nseg)
  xmin=[H*V]*nseg; xmax=[0]*nseg; 
  ymin=[H*V]*nseg; ymax=[0]*nseg; 
  for i in range(H*V):
    x=i%H; y=i//H; s=seg[i]
    if(x<xmin[s]): xmin[s]=x
    if(y<ymin[s]): ymin[s]=y
    if(x>xmax[s]): xmax[s]=x
    if(y>ymax[s]): ymax[s]=y
  center=np.zeros(nseg,dtype=np.uint32)
  for s in range(nseg):
    y=(ymin[s]+ymax[s])//2; x=(xmin[s]+xmax[s])//2; 
    center[s]=y*H+x
  return(center,nseg)

def select_training_samples_seg(truth,center,H,V,sizex,sizey,porcentaje):
  print('* Select training samples')
  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)
  for i in truth:
    if(i>nclases): nclases=i
  print('  nclasses:',nclases)
  lista=[0]*nclases;
  for i in range(nclases):
    lista[i]=[]
  xmin=int(sizex/2); xmax=H-int(math.ceil(sizex/2))
  ymin=int(sizey/2); ymax=V-int(math.ceil(sizey/2))
  for ind in center:
    i=ind//H; j=ind%H;
    if(i<ymin or i>ymax or j<xmin or j>xmax): continue
    if(truth[ind]>0): lista[truth[ind]-1].append(ind)
  for i in range(nclases):
    random.shuffle(lista[i])
  # seleccionamos muestras para train, validacion y test
  print('  Class  # :   total | train |   val |    test')
  train=[]; val=[]; test=[]
  for i in range(nclases):
    # tot0: numero muestras entrenamiento, tot1: validacion 
    if(porcentaje[0]>=1): tot0=porcentaje[0]
    else: tot0=int(porcentaje[0]*len(lista[i]))
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2
    if(tot0<0 and len(lista[i])>0): tot0=1
    if(tot0!=0): nclases_no_vacias+=1
    if(porcentaje[1]>=1): tot1=porcentaje[1]
    else: tot1=int(porcentaje[1]*len(lista[i]))
    if(tot1>=len(lista[i])-tot0): tot1=(len(lista[i])-tot0)//2
    if(tot1<1 and len(lista[i])>0): tot1=0
    for j in range(len(lista[i])):
      if(j<tot0): train.append(lista[i][j])
      elif(j<tot0+tot1): val.append(lista[i][j])
      # en test incluimos todos, luego seleccionaremos
      test.append(lista[i][j])
    print('  Class',f'{i+1:2d}',':',f'{len(lista[i]):7d}','|',f'{tot0:5d}','|',
      f'{tot1:5d}','|',f'{len(lista[i])-tot0-tot1:7d}')
  return(train,val,test,nclases,nclases_no_vacias)

#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------

# cogemos muestras sin ground-truth (dadas por el indice samples)
class HyperAllDataset(Dataset):
  def __init__(self,datos,samples,H,V,sizex,sizey):
    self.datos=datos; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose(
      [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    return(patch)

#----------------

# cogemos muestras con ground-truth (dadas por el indice samples)
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey):
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose(
      [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; truth=self.truth; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    # renumeramos porque la red clasifica tambien la clase 0 
    return(patch,truth[self.samples[idx]]-1)

#-----------------------------------------------------------------
# PYTORCH - UTIL
#-----------------------------------------------------------------

# pulsando CNLT-C acabamos el entrenamiento y pasamos a testear
def signal_handler(sig, frame):
  print('\n* Ctrl+C. Exit training')
  global endTrain
  endTrain=True

# For updating learning rate manual
def update_lr(optimizer,lr):    
  for param_group in optimizer.param_groups:
    param_group['lr']=lr

# calcula los promedios de precisiones
def accuracy_mean_deviation(OA,AA,aa):
  n=len(OA); nclases=len(aa[0])
  print('* Means and deviations (%d exp):'%(n))
  # medias
  OAm=0; AAm=0; aam=[0]*nclases;
  for i in range(n):
     OAm+=OA[i]; AAm+=AA[i]
     for j in range(1,nclases): aam[j]+=aa[i][j]
  OAm/=n; AAm/=n
  for j in range(1,nclases): aam[j]/=n
  # desviaciones, usamos la formula que divide entre (n-1)
  OAd=0; AAd=0; aad=[0]*nclases
  for i in range(n):
     OAd+=(OA[i]-OAm)*(OA[i]-OAm); AAd+=(AA[i]-AAm)*(AA[i]-OAm)
     for j in range(1,nclases): aad[j]+=(aa[i][j]-aam[j])*(aa[i][j]-aam[j])
  OAd=math.sqrt(OAd/(n-1)); AAd=math.sqrt(AAd/(n-1))
  for j in range(1,nclases): aad[j]=math.sqrt(aad[j]/(n-1))
  for j in range(1,nclases): print('  Class %02d: %02.02f+%02.02f'%(j,aam[j],aad[j]))
  print('  OA=%02.02f+%02.02f, AA=%02.02f+%02.02f'%(OAm,OAd,AAm,AAd))

#-----------------------------------------------------------------
# PYTORCH - NETWORK
#-----------------------------------------------------------------

import timm
import torch.nn as nn

#-----------------------------------------------------------------
# PYTORCH - MAIN
#-----------------------------------------------------------------

import timm
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
import argparse

#-----------------------------------------------------------------
# PYTORCH - MAIN
#-----------------------------------------------------------------


def obtener_modelo(model_name, pretrained=False, **kwargs):
    global model_registry
    if model_name in model_registry:
        return model_registry[model_name](pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is not registered in the registry.")


def main(exp):
  print('* FastVit exp: '+str(exp))
  time_start = time.time()
  # 1. Device configuration
  cuda = True if torch.cuda.is_available() else False
  print('* cuda:', cuda)
  device = torch.device('cuda' if cuda else 'cpu')
  if torch.backends.cudnn.is_available():
    print('* Activando CUDNN')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
  # experimentos deterministas o aleatorios
  if(DET == 1):
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if(cuda == False):
      torch.use_deterministic_algorithms(True)
      g = torch.Generator(); g.manual_seed(SEED)
    else:
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  # 2. Load datos
  (datos, H, V, B) = read_raw(DATASET)
  (truth, H1, V1) = read_pgm(GT)
  (seg, H2, V2) = read_seg(SEG)
  # necesitamos los datos en band-vector para hacer convoluciones
  datos = np.transpose(datos, (2, 0, 1))
  # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
  sizex = 32; sizey = 32 

  # 3. Selection training, testing sets
  (center, H3, V3, nseg) = read_seg_centers(CENTER)
  (train, val, test, nclases, nclases_no_vacias) = select_training_samples_seg(truth, center, H, V, sizex, sizey, SAMPLES)
  dataset_train = HyperDataset(datos, truth, train, H, V, sizex, sizey)
  print('  - train dataset:', len(dataset_train))
  dataset_test = HyperDataset(datos, truth, test, H, V, sizex, sizey)
  print('  - test dataset:', len(dataset_test))
  # Dataloader
  batch_size = BATCH # defecto 100
  train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
  test_loader = DataLoader(dataset_test, batch_size, shuffle=False)
  # Si queremos validacion
  if(len(val) > 0):
    dataset_val = HyperDataset(datos, truth, val, H, V, sizex, sizey)
    print('  - val dataset:', len(dataset_val))
    val_loader = DataLoader(dataset_val, batch_size, shuffle=False)
  
  # 4. Hyper parameters
  if(ADA == 0): 
    lr = args.lr
  else: 
    lr = args.lr
  
  # 5. Intancias del modelo del paquete timm
  # model = timm.create_model('fastvit_t8', pretrained=False, num_classes=nclases).to(device)
  # model = timm.create_model('fastvit_t16', pretrained=False, num_classes=nclases).to(device)
  # model = timm.create_model('fastvit_s12', pretrained=False, num_classes=nclases).to(device)
  # model = timm.create_model('fastvit_sa12', pretrained=False, num_classes=nclases).to(device)
  # model = timm.create_model('fastvit_sa24', pretrained=False, num_classes=nclases).to(device)
  # model = timm.create_model('fastvit_sa36', pretrained=False, num_classes=nclases).to(device)
  # model = timm.create_model('fastvit_ma36', pretrained=False, num_classes=nclases).to(device)

  # layers = [2, 2, 4, 2]
  # token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
  # embed_dims = [48, 96, 192, 384]
  # mlp_ratios = [3, 3, 3, 3]
  # downsamples = [True, True, True, True]

  # model = FastViT(
  #     layers=layers,
  #     token_mixers=token_mixers,
  #     embed_dims=embed_dims,
  #     mlp_ratios=mlp_ratios,
  #     downsamples=downsamples,
  #     num_classes=nclases,
  #     inference_mode=False
  # ).to(device)

  model = obtener_modelo(args.model, pretrained=False, num_classes=nclases).to(device)

  # Envuelve el modelo con DataParallel
  if torch.cuda.device_count() > 1:
    print(f'* Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)

  # 6. Loss, optimizer, and scheduler

  # 6.1 Define the loss function
  if args.loss == 'cross_entropy':
    criterion = nn.CrossEntropyLoss() # e o mais comun para tarefas de clasificacion multiclase con forte penalizacion das incorrectas
  elif args.loss == 'l1':
    criterion = nn.L1Loss()
  else:
    raise ValueError(f"A funcion de perda (loss) '{args.loss}' non esta soportada.")


  # 6.2 creamos un conjunto de optimizadores para los parámetros
  if args.optimizer == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
  elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
  elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
  elif args.optimizer == 'adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
  elif args.optimizer == 'adamax':
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
  elif args.optimizer == 'asgd':
    optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
  elif args.optimizer == 'lbfgs':
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
  else:
    raise ValueError(f"O optimizador '{args.optimizer}' non esta soportado.")

  # 6.3 scheduler (no es estrictamente necesario)
  if(ADA == 2): 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS//2, (5*EPOCHS)//6], gamma=0.1)
  elif(ADA == 3): 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0, verbose=True)
  elif(ADA == 4): 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, verbose=True)
  else: 
    pass

  # 7. Train the model

  # 7.1 Mixup

  # mixup_fn = Mixup(
  #     mixup_alpha=1.0,
  #     cutmix_alpha=0.0,
  #     cutmix_minmax=None,
  #     prob=1.0,
  #     switch_prob=0.5,
  #     mode='batch',
  #     label_smoothing=0.1,
  #     num_classes=nclases
  # )

  print('* Train FastViT, exp.%d' % (exp))
  global endTrain
  global mean_time
  time_total2 = []
  endTrain = False
  total_step = len(train_loader)
  
  for epoch in range(EPOCHS):
      time_start2 = time.time()
      for i, (inputs, labels) in enumerate(train_loader):
          if torch.isnan(inputs).any() or torch.isinf(inputs).any():
              print(f"Nan or Inf found in inputs at batch {i}")
          if torch.isnan(labels).any() or torch.isinf(labels).any():
              print(f"Nan or Inf found in labels at batch {i}")

          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
          if torch.isnan(outputs).any() or torch.isinf(outputs).any():
              print(f"Nan or Inf found in outputs at batch {i}")
              print("Outputs:", outputs)
              continue

          loss = criterion(outputs, labels)
          if torch.isnan(loss).any() or torch.isinf(loss).any():
              print(f"Nan or Inf found in loss at batch {i}")
              print("Loss:", loss)
              continue

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      
      time_end2 = time.time()
      time_total2.append(time_end2 - time_start2)

      if(epoch % 10 == 0 or epoch == EPOCHS - 1):
          print('  Tiempo por época (TPE) :', time_total2[epoch])
          if(len(val) > 0):
              model.eval()

              losses = []
              acces = []
              for i, (inputs, labels) in enumerate(val_loader):
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  outputs = model(inputs)

                  if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                      print(f"Nan or Inf found in validation inputs at batch {i}")
                  if torch.isnan(labels).any() or torch.isinf(labels).any():
                      print(f"Nan or Inf found in validation labels at batch {i}")
                  loss = criterion(outputs, labels)
                  acc = torch.mean((outputs.argmax(dim=-1) == labels).float())
                  losses.append(loss.item())
                  acces.append(acc.item())
              print('  Epoch: %3d/%d, Val. Loss: %.4f, Acc: %.4f' % (epoch, EPOCHS, sum(losses) / len(losses), sum(acces) / len(acces)))
          else:
              print('  Epoch: %3d/%d, Train Loss: %.4f' % (epoch, EPOCHS, loss.item()))

      if(ADA == 1 and (epoch + 1) % 20 == 0): lr /= 2; update_lr(optimizer, lr)
      elif(ADA > 1): scheduler.step()
      if(endTrain): break

  mean_time2 = sum(time_total2) / EPOCHS

  print('* Tiempo por epoca medio (TPEm) : %.2f s' % (mean_time2))

  if(TEST == 0): return(sum(acces) / len(acces))

  # 8. Test the model
  print('* Test FastViT, exp.%d' % (exp))
  output = np.zeros(H * V, dtype=np.uint8)
  model.eval()

  # model_inf = reparameterize_model(model)
  # model_inf = model_inf.to(device)
  # model_inf.eval()

  all_labels = []
  all_predictions = []
  with torch.no_grad():
    correct = 0; total = 0;
    for(inputs, labels) in test_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)

      (_, predicted) = torch.max(outputs.data, 1)
      predicted_cpu = predicted.cpu()
      all_labels.extend(labels.cpu().numpy())
      all_predictions.extend(predicted_cpu.numpy())
      for i in range(len(predicted_cpu)):
        output[test[total + i]] = np.uint8(predicted_cpu[i] + 1)
      total += labels.size(0)
      if(total % 2000 == 0):
        print('  Test: %6d/%d' % (total, len(dataset_test)))
  print('* Generating classif.map')
  for i in range(H * V): output[i] = output[center[seg[i]]]
  for i in train: output[i] = 0
  for i in val: output[i] = 0
  
  correct = 0; total = 0

  # Calcular la precisión a nivel de segmentos
  # Se considera correcto si el segmento central es correcto, es decir, el pixel central
  # El pixen central es el pixel en la posición center[seg[i]] y no
  # se asigna a ninguna clase
  
  for i in range(len(center)):
    if(output[center[i]] == 0): continue
    total += 1
    if(output[center[i]] == truth[center[i]]): correct += 1
  acc = 100 * correct / total;
  print('* Accuracy (segments): %.02f' % (acc))

  correct = 0; total = 0; AA = 0; OA = 0
  class_correct = [0] * (nclases + 1)
  class_total = [0] * (nclases + 1)
  class_aa = [0] * (nclases + 1)
  for i in range(len(output)):
    if(output[i] == 0 or truth[i] == 0): continue
    total += 1; class_total[truth[i]] += 1
    if(output[i] == truth[i]):
      correct += 1
      class_correct[truth[i]] += 1
  for i in range(1, nclases + 1):
    if(class_total[i] != 0): class_aa[i] = 100 * class_correct[i] / class_total[i]
    else: class_aa[i] = 0
    AA += class_aa[i]
  OA = 100 * correct / total; AA = AA / nclases_no_vacias 
  print('* Accuracy (pixels) exp.%d:' % (exp))
  for i in range(1, nclases + 1): print('  Class %02d: %02.02f' % (i, class_aa[i]))
  print('* Accuracy (pixels) exp.%d, OA=%02.02f, AA=%02.02f' % (exp, OA, AA))
  print('  total:', total, 'correct:', correct)

  # Calcular el coeficiente de Kappa
  kappa = cohen_kappa_score(all_labels, all_predictions)
  print('* Kappa (exp.%d): %.04f' % (exp, kappa))

  #save_pgm(output, H, V, nclases, args.output + '/output_FastVit-' + str(exp) + '.pgm')
  #torch.save(model.state_dict(), args.output + '/model_FastVit-' + str(exp) + '.ckpt')

  time_end = time.time()
  print('* Execution time: %.0f s' % (time_end - time_start))
  print('  lr:', lr, 'BATCH:', batch_size)
  return(OA, AA, class_aa, kappa)

if __name__ == '__main__':
  lista_resultados = []
  if(TEST == 0):
    acces = 0
    print(" Iniciando algoritmo . . .")
    for exp in range(EXP): acces = acces + main(exp)
    print('* FastVit SEG EXP:', EXP, 'EPOCHS:', EPOCHS, 'SAMPLES:', SAMPLES, 'ADA:', ADA, 'AUM:', AUM)
    print('  VAL: %02.02f' % (100 * acces / EXP))
  else:
    OA = [0] * EXP; AA = [0] * EXP; aa = [0] * EXP; kappa_scores = [0] * EXP
    for exp in range(EXP): 
      (OA[exp], AA[exp], aa[exp], kappa_scores[exp]) = main(exp)
      lista_resultados.append([exp ,OA[exp], AA[exp], aa[exp], kappa_scores[exp], ADA, mean_time])
    if(EXP > 1): accuracy_mean_deviation(OA, AA, aa) 
    print('* FastVit SEG EXP:', EXP, 'EPOCHS:', EPOCHS, 'SAMPLES:', SAMPLES, 'ADA:', ADA, 'AUM:', AUM)
    print('  Kappa: %.04f' % (sum(kappa_scores) / EXP))

    # Guardar resultados en un archivo
    with open(args.output + '/resultados.txt', 'w') as f:
      # Genera un encabezado con los nombres de las columnas
      f.write('EXP OA AA aa kappa_cohen ADA media_tiempo_epoch\n')
      for resultado in lista_resultados:
        f.write(f'{resultado}\n')
    print('* Resultados guardados en:', args.output + '/resultados.txt')

