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
    eiras
    ermidas
    mestas
    oitaven
    ulla
    xesta
)

for dataset in "${datasets[@]}"
do
    
#     echo "Adestrando modelo crossformer universal co dataset: $dataset"

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=96 \
            layers='2, 2, 2' \
            num-heads='3, 6, 12' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=96 \
            layers='2, 3, 3' \
            num-heads='2, 4, 8' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=96 \
            layers='4, 3, 2' \
            num-heads='3, 6, 12' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null


    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=96 \
            layers='2, 2, 2' \
            num-heads='2, 4, 8' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=96 \
            layers='4, 4, 2' \
            num-heads='2, 4, 8' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=48 \
            layers='2, 2, 2' \
            num-heads='3, 6, 12' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=crossformer_universal \
            patch-size=2 \
            embed-dim-inicial=48 \
            layers='2, 2, 2' \
            num-heads='2, 4, 8' \
            group-size='4, 2, 1' \
            mpl-ratio=4 \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

#     echo "Adestrando modelo coatnet universal co dataset: $dataset"

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=coatnet_universal \
            embed_dims='64, 96, 192, 384' \
            layers='1, 2, 2, 1' \
            block-types='C T C' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=coatnet_universal \
            embed_dims='64, 96, 192, 384' \
            layers='1, 1, 1, 1' \
            block-types='T T C' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=coatnet_universal \
            embed_dims='64, 96, 192, 384' \
            layers='2, 2, 1, 1' \
            block-types='T T C' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=coatnet_universal \
            embed_dims='64, 96, 192' \
            layers='1, 1, 1' \
            block-types='T T' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=coatnet_universal \
            embed_dims='64, 96, 192' \
            layers='1, 1, 1' \
            block-types='T C' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

    guild run modelos.py \
            dataset=./datasets/${dataset}_river.raw \
            gt=./datasets/${dataset}_river.pgm \
            seg=./datasets/seg_${dataset}_wp.raw \
            center=./datasets/seg_${dataset}_wp_centers.raw \
            model=coatnet_universal \
            embed_dims='64, 96, 192, 384' \
            layers='1, 1, 2, 2' \
            block-types='C T C' \
            optimizer=adam \
            batch=100 \
            --yes > /dev/null

done
