#!/bin/bash

local_rank=$(printenv OMPI_COMM_WORLD_LOCAL_RANK)
echo "$@"
case $local_rank in
  0)
    numa_node=0
    ;;
  1)
    numa_node=0
    ;;
  2)
    numa_node=1
    ;;
  3)
    numa_node=1
    ;;
  *)
    echo "Invalid local rank: $local_rank"
    exit 1
    ;;
esac

numactl --localalloc --cpunodebind=$numa_node "$@"
