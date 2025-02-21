import os
import sys
import itertools
import json

# Helper
def product(d):
    def list_or_id(item):
        if not isinstance(item, list):
            return [item]
        else:
            return item

    keys = list(d.keys())
    values = itertools.product(*[list_or_id(d[k]) for k in keys])

    for value in values:
        yield dict(zip(keys, value))

def to_args(params):
    return ' '.join([f'--{k} {v}' for k, v in params.items()])


# Parameters
project = 'debiasing-rag'
script = 'embedder/train.py'

args = {
    'seed': 0,
    'model_type': 'gte',
    'retriever_model': 'thenlper/gte-base', 
    'llm': 'meta-llama/Llama-3.1-8B-Instruct',
    'judge': 'gpt-4o-mini',
    'template': 'basic',
    'pooling': 'mean',
    'temperature': 50.0,
    'bias_dataset': ["GenderBias-QA"], 
    'query_dataset': 'rag-datasets/mini_wikipedia',
    'max_length': 512,
    'batch_size': 8, 
    'embedding_batch': 16,
    'loss': 'simcse', 
    'optimizer': 'adamw',
    'lr': [3e-5, 1e-5],
    'weight_decay': 0.01, 
    'scheduler': 'warmup_linear',
    'warmup_steps': 10,
    'epochs': [5, 10, 15], 
    'freeze_layers': [0, 1, 2, 3, 4],
    'merge_weight': [0.1, 0.3, 0.5, 0.7, 0.9]
} 

additional_args = {
    'base_dataset_dir': 'dataset/tasks/',
    'base_index_dir': 'dataset/filtered_bias_index/',
    'train_corpus': 'msmarco-fever-dbpedia', 
    # 'save_dir': 'models/embedders',
    'eval_save_dir': 'results/training',
    'cache_dir': os.environ.get('HUG_CACHE_DIR'),
    'save_prefix': 'layer',
    'chunk_size': 0,
    'num_devices': 1,
    'use_wandb': 0, 
    'wandb_project': 'Debiasing-RAG',
    'wandb_name': None,
    'eval_bias': 1,
    'eval_overwrite': 1,
    'rag_type': 'top1',
    'bias_set': 'train',
    'topk': 3,
    'top_p': 0.0,
}

launch_commands = []
for i, param in enumerate(product(dict(args, **additional_args))):
    id = f'{project}_' + '_'.join([str(param[k]) for k in param if k in args])
    id = id.replace('/', '_').replace('-', '_').replace('.', '_').replace(',', '_').replace(' ', '_').replace("'", '_').replace('"', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_')
    param['wandb_name'] = id
    launch_args = f'--num_processes {param["num_devices"]} --main_process_port `valid_port`'
    experiment_args = to_args(param)
    launch_command = f'accelerate launch {script} {experiment_args}'
    launch_commands.append(launch_command)
    
with open('training_command_lines.txt', 'w') as f:
    f.write('\n'.join(map(str, launch_commands)) + '\n')