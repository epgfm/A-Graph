#! /bin/bash

for i in {9..40}; do
    next=$(( $i + 1 ))
    ./compute_average_positions.py --final "final-$i.pkl.gz" --flist "flist-$next.pkl.gz"
    ./final_ablation_run.py --flist "flist-$next.pkl.gz" --final "final-$next.pkl.gz" --image "image-$next.png"
done
