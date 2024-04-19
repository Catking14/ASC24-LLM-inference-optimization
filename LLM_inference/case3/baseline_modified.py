"""Benchmark offline inference throughput."""
import argparse
import json
import random
import sys
from vllm import LLM, SamplingParams
from typing import List, Optional, Tuple
from mpi4py import MPI

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from transformers import BitsAndBytesConfig
from tqdm import tqdm
TOTAL_GPUS = 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_workload(dataset, gpus):
    # ceval = []
    prompts = []
    prompt_len_pile = [0] * gpus
    max_prompt_len = 0

    for i in range(len(dataset)):
        cur = f"{dataset['question'][i]}\n\nA. {dataset['A'][i]}\nB. {dataset['B'][i]}\nC. {dataset['C'][i]}\nD. {dataset['D'][i]}"
        select = prompt_len_pile.index(min(prompt_len_pile))

        if rank == select:
            prompts.append([cur, dataset["answer"][i]])

        max_prompt_len = max(max_prompt_len, len(cur))
        prompt_len_pile[select] += len(cur)

    print(f"data len for rank {rank} is {len(prompts)}.")

    return prompts, max_prompt_len, len(dataset)

# only produce batches with input prompts
def simple_batch(prompts, batch_size):
    batches = []

    for idx in range(0, len(prompts), batch_size):
        batch = []

        for i in range(batch_size):
            if idx + i < len(prompts):
                batch.append(prompts[idx + i][0])
            else:
                break
        
        batches.append(batch)

    return batches


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # initialize  model and tokenizer
    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

    # llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    #                 #    quantization_config=quantization_config, # Uncomment this line for 4bit quantization
    #                 )
    sampling_params = SamplingParams(temperature=1, top_p=1.0, max_tokens=1024, use_beam_search=True, best_of=1)
    llm = LLM(model="/work/AquilaChat2-34B-Int4-GPTQ", trust_remote_code=True, dtype="float16", quantization="gptq", tensor_parallel_size=2)

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    llm = llm.cuda()

    if args.dataset is None:
        print("dataset has to be set")
        sys.exit(1) 
    else:
        with open(args.dataset) as f:
            requests = json.load(f)
    
    if args.num_samples is not None:
        requests = requests[0:args.num_samples]
    
    data = []

    input_data = split_workload(requests, TOTAL_GPUS)
    batched_data = simple_batch(input_data, 16)

    # for i in tqdm(range(len(requests))):
    for i in tqdm(range(len(batched_data))):
        # prompt = requests[i]
        prompt = batched_data[i]
        # Generate the sequences.
        # input_ids = tokenizer(prompt, return_tensors="pt",
        #                       padding=True).input_ids
        # llm_outputs = llm.generate(
        #     input_ids=input_ids.cuda(),
        #     do_sample=False,
        #     num_beams=1,
        #     num_return_sequences=1,
        #     temperature=1.0,
        #     top_p=1.0,
        #     use_cache=True,
        #     max_new_tokens=1024,
        # )
        out = llm.generate(prompt, sampling_params=sampling_params)

        # decode
        # output= tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
        for gen_text in out:
            output = gen_text.outputs[0].text
            data.append({"input_text": prompt, "generated_text": output})
    
    # save to output.json
    json_file_path = args.output
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False) 
    print("The output file is saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for case 3.")
    parser.add_argument("--dataset", type=str, default="./data.json", help="Path to the dataset.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None, help="Number of first few samples used for inference test")
    parser.add_argument("--output", type=str, default="./output.json", help="Path to output file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)

