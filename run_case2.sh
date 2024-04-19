#!/bin/bash

GPUS_PER_NODE=2
TOTAL_PROCS=2
model_path=/work/AquilaChat2-34B-Int4-GPTQ
hosts=nthu1:1,nthu4:1

module purge
module load openmpi_gcc
# module load nvhpc-hpcx/24.1
module load cuda

# conda init
conda activate aquila

mpirun -np ${TOTAL_PROCS} -x LD_LIBRARY_PATH -x PATH -H ${hosts} bind_numa.sh lm_eval --tasks mmlu_flan_cot_fewshot --model vllm \
--model_args pretrained=${model_path},tensor_parallel_size=${GPUS_PER_NODE},dtype=float16,gpu_memory_utilization=0.8,data_parallel_size=1,trust_remote_code=True,max_gen_toks=1024,max_model_len=4096,quantization=gptq --batch_size 16 --num_fewshot 5 --gen_kwargs temperature=1,do_sample=False,num_beams=1 \
> out/case2/run.log 2>&1


