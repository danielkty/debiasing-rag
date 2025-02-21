import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import json
import copy

from accelerate import Accelerator

from embedder.trainer import EmbeddingTrainer
from embedder.data import JsonTaskMixtureDataset, HomogenousBatchMixtureSampler
from embedder.models import GteEmbedder
from embedder.evaluations import run_eval

def get_tasks_dataset_batch_loader(
        tasks,
        loss_type=None,
        transform=None,
        dataset_dir=None, 
        bias_set=None,
        index_dir_list=None,
        cache_dir=None,
        drop_last_each=False, 
        batch_size=None, 
        shuffle=True, 
        generator=None, 
        num_workers=0, 
        collate_fn=None,):
    if collate_fn is None:
        collate_fn = lambda batch: [[b[i] for b in batch] for i in range(len(batch[0]))]
    if transform is None:
        QUERY = 'query'
        DOCUMENT = 'document'
        transform = lambda data: (
            (QUERY, {'text': data['query']}),
            (DOCUMENT, {'text': data['pos']}),
            (DOCUMENT, {'text': data['neg']}))
    dataset = JsonTaskMixtureDataset(tasks, loss_type=loss_type, transform=transform, dataset_dir=dataset_dir, bias_set=bias_set, index_dir_list=index_dir_list, cache_dir=cache_dir, drop_last_each=drop_last_each, batch_size=batch_size)
    batch_sampler = HomogenousBatchMixtureSampler(dataset, batch_size, generator=generator, shuffle=shuffle)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
    loader.__dict__['batch_size'] = batch_size  # for accelerate
    return loader

def main(args):
    # Create accelerator
    accelerator = Accelerator(split_batches=True)

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed+1)
    np.random.seed(args.seed+2)

    # Initial wandb
    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

    # Checks
    if args.batch_size % args.num_devices != 0:
        raise ValueError(f'Batch size {args.batch_size} must be divisible by the number of devices {args.num_devices}.')
    if args.chunk_size != 0:
        if args.batch_size % args.chunk_size != 0:
            raise ValueError(f'Batch size {args.batch_size} must be divisible by the chunk size {args.chunk_size}.')

    # Load dataset
    dataset_generator = torch.Generator().manual_seed(args.seed+100)
    if args.bias_dataset == 'PoliticBias-QA':
        tasks = ['PoliticBias-QA']
    elif args.bias_dataset == 'GenderBias-QA':
        tasks = ['GenderBias-QA']
    else:
        raise ValueError(f'Invalid dataset: {args.bias_dataset}')

    index_dir_list = []
    corpuses = args.train_corpus.split('-')
    for filename in os.listdir(args.base_index_dir):
        if any(corpus in filename for corpus in corpuses) and all(element in filename for element in [args.bias_dataset, f"topk-{args.topk}", args.bias_set]):
            index_dir_list.append(os.path.join(args.base_index_dir, filename))

    dataset_dir = os.path.join(args.base_dataset_dir, args.bias_dataset)
    train_loader = get_tasks_dataset_batch_loader(
        tasks,
        loss_type=args.loss,
        dataset_dir=dataset_dir, 
        bias_set=args.bias_set,
        index_dir_list=index_dir_list,
        cache_dir=args.cache_dir,
        drop_last_each=True, 
        batch_size=args.batch_size, 
        shuffle=True, 
        generator=dataset_generator)

    model_path = os.path.join(args.model_dir, args.retriever_model) if args.model_dir is not None else args.retriever_model
    if args.model_type == 'gte':
        model = GteEmbedder(
            model_path=model_path,
            template=args.template, 
            pooling=args.pooling, 
            max_length=args.max_length,
            cache_dir=args.cache_dir)
    else:
        raise ValueError(f'Invalid model: {args.retriever_model}')


    if args.optimizer == 'adamw':
        optimizer_args = {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'betas': (0.9, 0.999),
        }
    else:
        raise ValueError(f'Invalid optimizer: {args.optimizer}')

    if args.scheduler == 'warmup_linear':
        scheduler_args = {
            'num_warmup_steps': args.warmup_steps,
            'num_training_steps': args.epochs * len(train_loader),
        }
    else:
        raise ValueError(f'Invalid scheduler: {args.scheduler}')

    if args.loss in ['simcse']:
        criterion_args = {
                'scale': args.temperature,
            }
    else:
        raise ValueError(f'Invalid loss: {args.loss}')
    
    if args.freeze_layers != 0:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.model.pooler.parameters():
            param.requires_grad = True

        for i in range(args.freeze_layers):
            layer_num = model.model.config.num_hidden_layers - 1 - i
            for param in model.model.encoder.layer[layer_num].intermediate.parameters():
                param.requires_grad = True
            for param in model.model.encoder.layer[layer_num].output.parameters():
                param.requires_grad = True

    if args.merge_weight != 0:
        orig_model = copy.deepcopy(model.model)

    trainer = EmbeddingTrainer(model)
    trainer.fit(
        train_loader=train_loader, 
        epochs=args.epochs,
        optimizer=args.optimizer, 
        optimizer_args=optimizer_args,
        criterion=args.loss, 
        criterion_args=criterion_args,
        scheduler=args.scheduler,
        scheduler_args=scheduler_args,
        grad_clip=1.,
        chunk_size=args.chunk_size,
        accelerator=accelerator,
        show_progress_bar=True,
        use_wandb=args.use_wandb,
    )

    if args.merge_weight != 0:
        orig_model.to(model.model.device)
        for p, q in zip(model.model.parameters(), orig_model.parameters()):
            p.data = q.data * (1 - args.merge_weight) + p.data * args.merge_weight

    if args.eval_bias:
        model.model.half()
        model.eval()
        bias_score = run_eval(model.model, model.tokenizer, args)

    if args.save_dir is not None:
        models_dir = os.path.join(args.eval_save_dir, f'{args.bias_dataset}/{args.train_corpus}/top-{args.topk}/{args.retriever_model.split("/")[-1]}')
        for filename in os.listdir(models_dir):
            with open(os.path.join(models_dir, filename), 'r') as f:
                data = json.load(f)
            arg_dict = data['arg_info']
            if arg_dict['lr'] == args.lr and arg_dict['epochs'] == args.epochs and arg_dict['freeze_layers'] == args.freeze_layers and arg_dict['merge_weight'] == args.merge_weight:
                bias_score = round(data['bias'], 1)
                break

        model_path = f"{args.bias_dataset.split('/')[-1]}/{args.retriever_model.split('/')[-1]}-{args.train_corpus}-top-{args.topk}-{bias_score}-lr-{args.lr}-e-{args.epochs}-l-{args.freeze_layers}-m-{args.merge_weight}"
        save_path = os.path.join(args.save_dir, model_path)
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        with open(save_path + '/args_config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)

    # Model
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--retriever_model', type=str)
    parser.add_argument('--llm', type=str)
    parser.add_argument('--judge', type=str)
    parser.add_argument('--template', type=str)
    parser.add_argument('--pooling', type=str)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--merge_weight', type=float)
    
    # Data
    parser.add_argument('--base_dataset_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--base_index_dir', type=str)
    parser.add_argument('--eval_save_dir', type=str)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--train_corpus', type=str)
    parser.add_argument('--query_dataset', type=str)
    parser.add_argument('--bias_dataset', type=str)
    parser.add_argument('--topk', type=str)
    parser.add_argument('--synthetic_corpus', type=str)

    # Training
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--freeze_layers', type=int)
    parser.add_argument('--embedding_batch', type=int)

    # Evaluation
    parser.add_argument('--rag_type', type=str)
    parser.add_argument('--bias_set', type=str)
    parser.add_argument('--top_p', type=float)

    # Optimizer
    parser.add_argument('--loss', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--epochs', type=int)

    # System info
    parser.add_argument('--num_devices', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--use_wandb', type=int)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--eval_bias', type=int)
    parser.add_argument('--eval_overwrite', type=int)
    parser.add_argument('--save_prefix', type=str)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 1000000000)

    main(args)