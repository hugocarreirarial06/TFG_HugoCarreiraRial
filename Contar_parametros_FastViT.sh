#!/bin/bash

datasets=(
    ferreiras
)

for dataset in "${datasets[@]}"
do
    
    echo "Adestrando modelo fastvit universal co dataset: $dataset"

     guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='64, 128' \
                layers='2, 6' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                optimizer=adam \
                batch=100 \
                --yes > /dev/null

    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='48, 96' \
                layers='1, 1' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                optimizer=adam \
                batch=100 \
                --yes > /dev/null


    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='64, 128' \
                layers='3, 2' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                optimizer=adam \
                batch=100 \
                --yes > /dev/null

    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='48, 96' \
                layers='1, 1' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                optimizer=adam \
                batch=100 \
                --yes > /dev/null

    # guild run contarParametros.py \
    #             dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
    #             gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
    #             seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
    #             center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
    #             model=fastvit_universal \
    #             embed_dims='48, 96' \
    #             layers='1, 2' \
    #             mlp_ratios='4, 4' \
    #             token-mixers='attention,repmixer' \
    #             optimizer=adam \
    #             batch=100 \
    #             --yes > /dev/null

    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='48, 96' \
                layers='2, 2' \
                mlp_ratios='3, 3' \
                token-mixers='repmixer,repmixer' \
                --yes > /dev/null

    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='64, 128' \
                layers='1, 1' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                --yes > /dev/null


     guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='64, 128' \
                layers='2, 6' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                --yes > /dev/null

    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='48, 96' \
                layers='2, 2' \
                mlp_ratios='3, 3' \
                token-mixers='repmixer,attention' \
                --yes > /dev/null

    guild run contarParametros.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='64, 128' \
                layers='3, 2' \
                mlp_ratios='4, 4' \
                token-mixers='repmixer,repmixer' \
                --yes > /dev/null

done
