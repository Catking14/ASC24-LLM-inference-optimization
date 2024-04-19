"""Benchmark offline inference throughput."""
import argparse
import json
import random
import sys
from typing import List, Optional, Tuple

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from transformers import BitsAndBytesConfig
from tqdm import tqdm


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

    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                    #    quantization_config=quantization_config, # Uncomment this line for 4bit quantization
                    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
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

    for i in tqdm(range(len(requests))):
        prompt = requests[i]
        # Generate the sequences.
        input_ids = tokenizer(prompt, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=1024,
        )
        # decode
        output= tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
        
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

