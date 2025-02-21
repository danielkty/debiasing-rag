from datasets import load_dataset, load_from_disk
import torch
from torch import Tensor
from torch.nn.functional import normalize
import os
import json
from tqdm import tqdm
import sys
from src.score_utils import get_correctness_score, get_bias_score, get_topics
from src.inference_utils import get_llm, get_predictions

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_acc_dataset(args):
    corpus = load_dataset(args.query_dataset, "text-corpus")
    corpus = corpus["passages"]["passage"]
    
    queries = load_dataset(args.query_dataset, "question-answer")
    answers = queries["test"]["answer"]
    queries = queries["test"]["question"]
    
    return queries, corpus, answers

def get_bias_dataset(args):
    file_path = f'dataset/tasks/{args.bias_dataset}'
    dataset = load_from_disk(file_path)
    corpus = dataset["corpus"]
    corpus = corpus["text"]
    dataset = dataset[args.bias_set]
    queries = dataset["queries"]

    return queries, corpus

def get_dataset_embeddings_index(retriever_model, retriever_tokenizer, args):
    acc_queries, acc_corpus, acc_answers = get_acc_dataset(args)
    bias_queries, bias_corpus = get_bias_dataset(args)
    acc_query_embeddings, acc_corpus_embeddings = get_embeddings(retriever_model, retriever_tokenizer, acc_queries, acc_corpus, args)
    bias_query_embeddings, bias_corpus_embeddings = get_embeddings(retriever_model, retriever_tokenizer, bias_queries, bias_corpus, args)
    acc_index = get_index(acc_query_embeddings, acc_corpus_embeddings)
    bias_index = get_index(bias_query_embeddings, bias_corpus_embeddings)
    
    return acc_queries, acc_corpus, acc_answers, bias_corpus, acc_index, bias_index

def get_bias_acc(acc_predictions, acc_answers, acc_queries, bias_corpus, args):
    acc_scores = get_correctness_score(acc_predictions, acc_answers, acc_queries, args.judge)
    bias_topic1, bias_topic2 = get_topics(args.bias_dataset)

    bias1_scores = get_bias_score(bias_corpus, topic=bias_topic1, judge=args.judge)
    bias2_scores = get_bias_score(bias_corpus, topic=bias_topic2, judge=args.judge)

    bias = round(sum(bias1_scores) / len(bias1_scores) * 100, 1) - round(sum(bias2_scores) / len(bias2_scores) * 100, 1)
    acc = round(sum(acc_scores) / len(acc_scores) * 100, 1)
    return acc, bias

def get_embeddings(retriever_model, retriever_tokenizer, queries, corpus, args):
    print(f"\nGenerating embeddings\n")
    corpus_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), args.embedding_batch), file=sys.stdout):
            input_corpus = corpus[i:i+args.embedding_batch]

            if args.model_type == 'e5':
                input_corpus = [f'passage: {text}' for text in input_corpus]
            
            batch_dict = retriever_tokenizer(input_corpus, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            corpus_embeddings.append(batch_embeddings.cpu())

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    
    query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), args.embedding_batch), file=sys.stdout):
            input_queries = queries[i:i+args.embedding_batch]

            if args.model_type == 'e5':
                input_queries = [f'query: {text}' for text in input_queries]

            batch_dict = retriever_tokenizer(input_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            query_embeddings.append(batch_embeddings.cpu())

    query_embeddings = torch.cat(query_embeddings, dim=0)
        
    return query_embeddings, corpus_embeddings

def get_index(query_embeddings, corpus_embeddings):
    norm_q_embeddings = normalize(query_embeddings, p=2, dim=1)
    norm_c_embeddings = normalize(corpus_embeddings, p=2, dim=1)
    scores = (norm_q_embeddings @ norm_c_embeddings.T)
    index = scores.argmax(dim=1)

    return index.reshape(-1, 1)

def save_scores(accuracy, bias, arg_info):
    save_dir = os.path.join(arg_info.eval_save_dir, f'{arg_info.bias_dataset}/{arg_info.train_corpus}/top-{arg_info.topk}/{arg_info.retriever_model.split("/")[-1]}')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exist_flag = False
    for exist_name in os.listdir(save_dir):
        exist_path = os.path.join(save_dir, exist_name)
        with open(exist_path, 'r') as f:
            data = json.load(f)
        arg_dict = data['arg_info']
        if arg_dict == vars(arg_info):
            print(f"File already exists: {exist_name}")
            exist_flag = True
            break
        
    save_data = {
        'accuracy': accuracy,
        'bias': bias,
        'arg_info': vars(arg_info)
    }
        
    if exist_flag:
        if arg_info.eval_overwrite:
            print(f"Overwriting file: {exist_name}")
            save_data['name'] = exist_name
            with open(exist_path, 'w') as f:
                json.dump(save_data, f)
    else:
        number = 1
        while True:
            save_name = f"model_{arg_info.save_prefix}_{number}.json"
            save_path = os.path.join(save_dir, save_name)
            if not os.path.exists(save_path):
                print(f"Saving file: {save_name}")
                save_data['name'] = save_name
                with open(save_path, 'w') as f:
                    json.dump(save_data, f)
                break
            number += 1

def run_eval(retriever_model, retriever_tokenizer, arg_info):
    acc_queries, acc_corpus, acc_answers, bias_corpus, acc_index, bias_index = get_dataset_embeddings_index(retriever_model, retriever_tokenizer, arg_info)
    del retriever_model, retriever_tokenizer

    llm_model, llm_tokenizer = get_llm(arg_info)
    acc_predictions = get_predictions(llm_model, llm_tokenizer, acc_queries, acc_corpus, acc_index, None, arg_info)
    del llm_model, llm_tokenizer

    selected_corpus = [bias_corpus[i] for i in bias_index.flatten()]
    accuracy, bias = get_bias_acc(acc_predictions, acc_answers, acc_queries, selected_corpus, arg_info)
    
    print(f"\nAccuracy: {accuracy}, Embedder Bias: {bias}\n")
    save_scores(accuracy, bias, arg_info)
    
    return bias
    
if __name__ == "__main__":
    pass