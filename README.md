# ASC24-LLM inference optimization

The code and scripts used in ASC24 LLM inference optimization challenge for National Tsing Hua University Team. 

## Challenge description

This task focuses on the LLM inference optimization with AquilaChat2-34B model, which requires participating teams to build an inference engine with 4 bit or lower quantization to achieve high throughput while maintaining the quantization error within 1.5% on the MMLU dataset provided by the ASC24 Committees. 

## Quantization Scripts

In this task, we use *Marlin* + GPTQ with 4 bit quantization. To retrieve quantized AquilaChat2-34B with optimized GPTQ, use the modified GPTQ quantize script `./marlin/gptq/llama2.py`, which is a fork from the *Marlin* repo. Check out [Marlin](https://github.com/IST-DASLab/marlin) for more information.

## Run Scripts

The final task requirement given on site is to run inference with MMLU dataset by *lm-evaluation-harness* module. The `lm-eval` executable can be installed via pip or install from source. Check [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for more.

To run the required task, execute `run_case$(i).sh` directly, where `$(i)` indicated the case you want to run.

> **Note: remember to modify `model_path` and `hosts` in the run scripts to match your system. `baseline_modified.py` is inside `./LLM_inference/case3` folder.**

## Other files

there are some other files which are used during preparation and experiments.

### `preparation-scripts`

The `preparation-scripts` folder stores the scripts and optimization attempts we used before contest. Since the final task and dataset is given on site, we use `ceval/ceval-exam` dataset for benchmarking and testing.

### `prometheus`

This folder stores the scripts for monitoring the cluster status with `prometheus`. We use `ipmitool` to monitor CPU power usage and temperature. `nvidia-smi` and *DCGM-Exporter* is used for monitoring GPU power and temperature.

## Output submission 

The final submission for the contest on site is backed up in `./LLM_inference`.
