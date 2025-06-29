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
    xesta
)

for dataset in "${datasets[@]}"
do
    
    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='48, 96, 192' \
            layers='3, 2, 1' \
            sr-ratios='2, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='48, 96, 192' \
            layers='4, 3, 2' \
            sr-ratios='2, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='48, 96, 192' \
            layers='2, 2, 1' \
            sr-ratios='2, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null


    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192, 384' \
            layers='1, 1, 1' \
            sr-ratios='4, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192, 384' \
            layers='3, 2, 1' \
            sr-ratios='1, 2, 4' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null


    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192, 384' \
            layers='3, 2, 1' \
            sr-ratios='2, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192, 384' \
            layers='2, 2, 1' \
            sr-ratios='2, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192, 384' \
            layers='1, 1, 1' \
            sr-ratios='2, 2, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192, 384' \
            layers='1, 1, 1' \
            sr-ratios='1, 2, 4' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='48, 96' \
            layers='3, 2' \
            sr-ratios='2, 2' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='48, 96' \
            layers='2, 2' \
            sr-ratios='2, 2' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='48, 96' \
            layers='1, 1' \
            sr-ratios='4, 4' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192' \
            layers='1, 1' \
            sr-ratios='1, 2' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192' \
            layers='3, 2' \
            sr-ratios='2, 2' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.raw \
            gt=/home/hugo.carreira/TFG/guildai/datasets/${dataset}_river.pgm \
            seg=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp.raw \
            center=/home/hugo.carreira/TFG/guildai/datasets/seg_${dataset}_wp_centers.raw \
            model=twins_universal \
            patch-size=2 \
            embed_dims='96, 192' \
            layers='2, 2' \
            sr-ratios='1, 1' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null
done
