#!/bin/bash

type="glove"

input_file="unigarment/collect/collect_original_pcd/usd_path_txt/gloves/glove_qualified.txt"

i=0
while IFS= read -r garment_usd_path; do
    echo "Collecting data for index $i, path $garment_usd_path"
    /home/user/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh unigarment/collect/collect_original_pcd/collect_pcd.py "$type" "$i" "$garment_usd_path"
    i=$((i + 1))
done < "$input_file"

