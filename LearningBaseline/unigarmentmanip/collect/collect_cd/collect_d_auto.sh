#!/bin/bash

type="bowl_hat"

input_file="unigarment/collect/collect_cd/prepare/bowl_hat/bowl_hat.txt"

i=0
while IFS= read -r garment_usd_path; do
    

    for j in {1..2}; do
        echo "Collecting data for index $i, path $garment_usd_path"

        timeout 90 /home/user/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh unigarment/collect/collect_cd/collect_hat_deformation.py "$type" "$i" "$garment_usd_path"
        
        # 检查 timeout 是否成功
        if [[ $? -eq 124 ]]; then
            echo "Process for index $i, iteration $j timed out and was skipped"
        fi
    done
    i=$((i + 1))
done < "$input_file"
