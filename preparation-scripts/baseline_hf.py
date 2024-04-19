from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

from datasets import load_dataset, concatenate_datasets
import time

device = torch.device("cuda")
# model_info = "BAAI/AquilaChat2-7B"
model_path = "/mlsteam/data/data/AquilaChat2-34B"

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
ceval = []

for i in range(10):
    temp = load_dataset("ceval/ceval-exam", name=ceval_subset_name[i], split="val")
    ceval.append(temp)

dataset = concatenate_datasets(ceval)
# print(dataset)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, quantization_config=quantization_config)
# print(model)
model.eval()

total_prompts_len = len(dataset)
correct = 0

begin = time.time()

# from predict import predict
for text in dataset:
    input_ids = tokenizer(f"{text['question']}\n\nA. {text['A']}\nB. {text['B']}\nC. {text['C']}\nD. {text['D']}", 
                          return_tensors="pt")["input_ids"].to(device)
    out = model.generate(input_ids=input_ids, max_length=300)
    out_str = tokenizer.decode(out[0])
    # print(f"#{text['id']}.\n {out_str}")

    ans = out_str.splitlines()

    if text["answer"] in ans[-1]:
        correct += 1

print(f"Total execution time is :{time.time() - begin} sec.")
print(f"prediction accuracy is {correct / total_prompts_len * 100} % in {total_prompts_len} prompts.")
