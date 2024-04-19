from mpi4py import MPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from vllm import LLM, SamplingParams
import torch

import time
import os
from datasets import load_dataset, concatenate_datasets
TOTAL_GPUS = 8

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# dataset
ceval_subset_name = ['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine',
                      'business_administration', 'chinese_language_and_literature', 'civil_servant',
                      'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics',
                      'college_programming', 'computer_architecture', 'computer_network', 'discrete_mathematics', 
                      'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer', 'fire_engineer', 
                      'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history', 
                      'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'ideological_and_moral_cultivation', 'law', 
                      'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry', 
                      'middle_school_geography', 'middle_school_history', 'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics', 
                      'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 'probability_and_statistics', 'professional_tour_guide', 
                      'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine']

def split_workload(dataset_name, gpus):
    ceval = []
    prompts = []
    prompt_len_pile = [0] * gpus
    max_prompt_len = 0

    for i in range(50):
        temp = load_dataset(dataset_name, name=ceval_subset_name[i], split="val")
        ceval.append(temp)

    dataset = concatenate_datasets(ceval)

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

def model_generate(device):
    # load dataset
    dataset, max_length, total_data_len = split_workload("ceval/ceval-exam", TOTAL_GPUS)
    # ceval = []
    # prompts = []

    # for i in range(10):
    #     temp = load_dataset("ceval/ceval-exam", name=ceval_subset_name[i], split="val")
    #     ceval.append(temp)

    # dataset = concatenate_datasets(ceval)
    # for i in range(len(dataset)):
    #     cur = f"{dataset['question'][i]}\n\nA. {dataset['A'][i]}\nB. {dataset['B'][i]}\nC. {dataset['C'][i]}\nD. {dataset['D'][i]}"
    #     prompts.append([cur, dataset["answer"][i]])
    batched_prompts = simple_batch(dataset, 64)
    print("=========== Dataset Loaded ===========")

    model_path = "/opt/AquilaChat/aquilachat2-34b"

    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bnb=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                        )

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, quantization_config=bnb)
    # model.eval()

    sampling_params = SamplingParams(temperature=0.9, top_p=0.95)
    llm = LLM(model="/opt/AquilaChat/AquilaChat2-34B-Int4-GPTQ", trust_remote_code=True, dtype="float16", quantization="gptq", local_rank=device)
    # llm = LLM(model="/opt/AquilaChat/aquilachat2-34b", trust_remote_code=True, dtype="float16", auto_quant_mode="weight_int4")

    correct = 0

    # sync all gpu devices
    # acc.wait_for_everyone()
    begin = time.time()
    result = []

    print("============ Start Generation =============")

    for text in batched_prompts:
        out = llm.generate(text, sampling_params=sampling_params)

        # if text[1] in out[0].outputs[0].text:
        #     correct += 1

        # result.append(out[0].outputs[0].text)
        for output_text in out:
            print(output_text.outputs[0].text)

    rank_time = time.time() - begin

    print(f"Total time for rank {rank} is {rank_time}.")

    comm.barrier()
    total_time = comm.reduce(rank_time, op=MPI.MAX, root=0)
    # correct_total = comm.reduce(correct, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Total time is {total_time} in {total_data_len} prompts.")
        print(f"Total prediction accuracy is {correct / len(dataset) * 100}.")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(add_help=False)

    # parser.add_argument(
    #     "local_rank",
    #     type=str
    # )

    # args = parser.parse_args()
    # args.device = torch.device(f"cuda:{args.local_rank}")
    local_rank = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    # device_for_rank = torch.device(f"cuda:{local_rank}")

    print("========== Start Inference ===========")
    model_generate(device=local_rank)


