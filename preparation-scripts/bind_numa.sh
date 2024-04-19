#!/bin/bash

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK}

echo "$@"
echo $local_rank
# case $local_rank in
#     0)
#         CUDA_VISIBLE_DEVICES=0 ;;
#     1)
#         CUDA_VISIBLE_DEVICES=1 ;;
# esac

case $local_rank in
    0)
        numa_node=0 ;;
    1)
        numa_node=1 ;;
esac

# export CUDA_VISIBLE_DEVICES
export PATH=/opt/anaconda3/envs/aquila/bin:$PATH
numactl --preferred=$numa_node --cpunodebind=$numa_node "$@"
