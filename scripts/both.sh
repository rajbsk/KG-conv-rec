#!/bin/bash
graph=$1
num_runs=$2
gpu_id=$3

if [ $graph == "2_hop" ]; then
    for i in $(seq 0 $((num_runs-1)));
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 parlai/tasks/redial/train_kbrd_2.py -mf saved/both_rgcn_$i
    done
fi

if [ $graph == "3_hop" ]; then
    for i in $(seq 0 $((num_runs-1)));
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 parlai/tasks/redial/train_kbrd_3.py -mf saved/both_rgcn_$i
    done
fi

if [ $graph == "5_hop" ]; then
    for i in $(seq 0 $((num_runs-1)));
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 parlai/tasks/redial/train_kbrd_5.py -mf saved/both_rgcn_$i
    done
fi

if [ $graph == "pr" ]; then
    for i in $(seq 0 $((num_runs-1)));
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 parlai/tasks/redial/train_kbrd_pp.py -mf saved/both_rgcn_$i
    done
fi

if [ $graph == "pr_0.9" ]; then
    for i in $(seq 0 $((num_runs-1)));
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 parlai/tasks/redial/train_kbrd_pp9.py -mf saved/both_rgcn_$i
    done
fi

if [ $graph == "pr_0.7" ]; then
    for i in $(seq 0 $((num_runs-1)));
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 parlai/tasks/redial/train_kbrd_pp7.py -mf saved/both_rgcn_$i
    done
fi
