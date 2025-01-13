#!/bin/bash

type="trousers"
input_file="unigarment/collect/collect_cd/prepare/trousers/short/short_trousers.txt"

i=223
while IFS= read -r garment_usd_path; do

    for j in {0..4}; do
        echo "Collecting data for index $i, path $garment_usd_path"

        # 根据 j 的值设置旋转角度
        if [ "$j" -eq 0 ]; then
            rotate_x=40
            rotate_y=0
        elif [ "$j" -eq 1 ]; then
            rotate_x=45
            rotate_y=0
        elif [ "$j" -eq 2 ]; then
            rotate_x=50
            rotate_y=0
        elif [ "$j" -eq 3 ]; then
            rotate_x=55
            rotate_y=0
        elif [ "$j" -eq 4 ]; then
            rotate_x=60
            rotate_y=0
        fi

        # 执行 Python 脚本，使用 timeout 防止超时
        timeout 90 /home/user/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh unigarment/collect/collect_cd/collect_lean_deformation.py "$type" "$i" "$garment_usd_path" "$rotate_x" "$rotate_y"
        exit_code=$?
        
        # 检查 timeout 是否成功
        if [ $exit_code -eq 124 ]; then
            echo "Process for index $i, iteration $j timed out and was skipped"
        elif [ $exit_code -ne 0 ]; then
            echo "Process for index $i, iteration $j failed with exit code $exit_code"
        fi
    done
    i=$((i + 1))
done < "$input_file"
/