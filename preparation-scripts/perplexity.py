# from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from datasets import load_dataset, concatenate_datasets

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
prompts = []

for i in range(10):
    temp = load_dataset("ceval/ceval-exam", name=ceval_subset_name[i], split="val")
    ceval.append(temp)

dataset = concatenate_datasets(ceval)

perplexity = load("perplexity", module_type="metric")
model_path = "/work/AquilaChat2-34B"
# model = AutoModelForCausalLM.from_pretrained(model_path)

input_texts = [s for s in dataset["question"] if s!='']
results = perplexity.compute(predictions=input_texts, model_id=model_path)

print(result["mean_perplexity"])