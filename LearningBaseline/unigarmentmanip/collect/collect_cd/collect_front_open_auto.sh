#!/bin/bash

type="front_open"
input_file="unigarment/collect/collect_cd/prepare/front_open/front_open.txt"

i=0
while IFS= read -r garment_usd_path; do

    for j in {1..25}; do
        echo "Collecting data for index $i, path $garment_usd_path"

        # 根据 j 的值设置旋转角度
        if [ $j -le 5 ]; then
            rotate_x=0
            rotate_y=0
        elif [ $j -le 10 ]; then
            rotate_x=30
            rotate_y=0
        elif [ $j -le 15 ]; then
            rotate_x=45
            rotate_y=0
        elif [ $j -le 20 ]; then
            rotate_x=0
            rotate_y=30
        else
            rotate_x=0
            rotate_y=45
        fi

        # 执行 Python 脚本，使用 timeout 防止超时
        timeout 90 /home/user/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh unigarment/collect/collect_cd/collect_front_open_deformation.py "$type" "$i" "$garment_usd_path" "$rotate_x" "$rotate_y"
        
        # 检查 timeout 是否成功
        if [[ $? -eq 124 ]]; then
            echo "Process for index $i, iteration $j timed out and was skipped"
        fi
    done
    i=$((i + 1))
done < "$input_file"
