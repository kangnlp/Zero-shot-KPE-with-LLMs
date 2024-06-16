import os
import re
import json
import argparse
from tqdm import tqdm
import datetime
from stanfordcorenlp import StanfordCoreNLP
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random



def get_generated_output(str):
    split_prompt_output = str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
    return split_prompt_output[-1].strip().replace("<|eot_id|>", "")



if __name__ == '__main__':

    task_instruction = "You are a keyphrase extractor. Extract keyphrases from the text. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase N'"
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path")
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets")
    parser.add_argument('--task_instruction', type=str, default=task_instruction, help="Vanilla prompt")
    parser.add_argument('--max_new_tokens', type=str, default='128', help="Input Document Max Length")
    parser.add_argument('--cuda', type=str, default='0', help="GPU")
    parser.add_argument('--core_nlp_path', type=str, default='stanford-corenlp-full-2018-02-27', help="Your StanfordCoreNLP path")
    parser.add_argument('--auth_token', type=str, default='', help="auth_token for Llama")

    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    task_instruction = args.task_instruction
    max_new_tokens = int(args.max_new_tokens)
    other_max_new_tokens = 512
    StanfordCoreNLP_path = args.core_nlp_path
    auth_token = args.auth_token


    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, output_attentions=True, token=auth_token)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    model.generation_config

    settings = {
        'model_name': model_name,
        'task_instruction': task_instruction,
        'max_new_tokens': max_new_tokens,
        'tokenizer_max_len': 4096,
        'max_len': 512,
        'do_sample': False
    }

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join('results', model_name.split('/')[-1], f'{timestamp}_vanilla')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")
    
    settings_path = os.path.join(result_path, 'settings.json')
    with open(settings_path, 'w') as settings_file:
        json.dump(settings, settings_file, indent=4)

    print(f"Settings saved to {settings_path}")

    en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)

    dataset_list = ['Inspec', 'SemEval2017', 'SemEval2010', 'DUC2001', 'nus', 'krapivin'] 

    for dataset_name in dataset_list:
            
        file_path = os.path.join(data_path,f'{dataset_name}_MAX512.jsonl')

        print(f"Dataset: {dataset_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines  = f.readlines()
            data_list = [json.loads(line.strip()) for line in lines]


        output_list = []

        for j_data in tqdm(data_list):

            doc = j_data['doc']
            prompt = prompt_template.format(task_instruction, doc)

            inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)

            with torch.no_grad():
                outputs = model.generate(**inputs.to(device), max_new_tokens=max_new_tokens, use_cache=True, do_sample=False, top_p=None, temperature=None )
            outputs_str = tokenizer.decode(outputs[0])

            generated_output_str = get_generated_output(outputs_str)
            #print(generated_output_str)

            pred_keyphrases_seq = generated_output_str.lower().split('keyphrases:')[-1].strip().rstrip('.')
            pred_keyphrases_list = [ pred.strip() for pred in pred_keyphrases_seq.split(';') ]


            log = {}
            log['prompt'] = prompt
            log['generated_ourput'] = generated_output_str
            log['final_pred_keyphrase'] = pred_keyphrases_list
            log['doc'] = doc
            log['label'] = j_data['label']
            log['stemmed_label'] = j_data['stemmed_label']

            output_list.append(log)

        file_path = os.path.join(result_path, f'{dataset_name}_result.json')
        with open(file_path, "w", encoding='utf-8') as f:
            for json_data in output_list:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
