#!/bin/bash
#SBATCH --job-name=traballo-fin-grao-hugo-carreira
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=72:00:00
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --output=creacion_pacthes_fastvit2_%j.log
#SBATCH -e creacion_pacthes_fastvit2_%j.err  # <-- Separa errores

# Lista de datasets
datasets=(
    ferreiras
)

for dataset in "${datasets[@]}"
do
    
    guild run contarParametros.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=swin_universal \
            embed-dim-inicial=96 \
            layers='2, 2' \
            num-heads='2, 2' \
            mlp-ratio=2 \
            window-size=4 \
            patch-size=2 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null


    guild run contarParametros.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=swin_universal \
            embed-dim-inicial=96 \
            layers='3, 6' \
            num-heads='3, 6' \
            mlp-ratio=2 \
            window-size=4 \
            patch-size=2 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run contarParametros.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=swin_universal \
            embed-dim-inicial=96 \
            layers='3, 6' \
            num-heads='3, 6' \
            mlp-ratio=4 \
            window-size=4 \
            patch-size=2 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run contarParametros.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=swin_universal \
            embed-dim-inicial=96 \
            layers='2, 2' \
            num-heads='2, 2' \
            mlp-ratio=4 \
            window-size=4 \
            patch-size=2 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run contarParametros.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=swin_universal \
            embed-dim-inicial=96 \
            layers='1, 1' \
            num-heads='3, 6' \
            mlp-ratio=2 \
            window-size=4 \
            patch-size=2 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null           

    guild run contarParametros.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=swin_universal \
            embed-dim-inicial=96 \
            layers='1, 1' \
            num-heads='1, 1' \
            mlp-ratio=2 \
            window-size=4 \
            patch-size=2 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null   

done
