#!/bin/bash
#SBATCH --job-name=patches-hugo-tfg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --output=creacion_pacthes_fastvit2_%j.log
#SBATCH -e creacion_pacthes_fastvit2_%j.err  # <-- Separa errores

# Lista de datasets
# datasets=(
#     eiras
# )

datasets=(
    xesta
)

for dataset in "${datasets[@]}"
do
    
    echo "Adestrando modelo fastvit universal co dataset: $dataset"

     guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
                dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
                gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
                seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
                center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
                model=fastvit_universal \
                embed_dims='48, 96' \
                layers='1, 2' \
                mlp_ratios='4, 4' \
                token-mixers='attention,repmixer' \
                optimizer=adam \
                batch=100 \
                --yes > /dev/null

    guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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


     guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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

    guild run modelo_fastvit.py \
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
