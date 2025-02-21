import argparse
import torch
from torch import Tensor
from torch.nn.functional import normalize
import os
import json
from tqdm import tqdm
from score_utils import get_bias_score, get_topics
from data_utils import get_bias_dataset
from inference_utils import get_predictions, get_retriever, get_llm
import numpy as np

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(retriever_model, retriever_tokenizer, queries, corpus):
    retriever_model.eval()
    print(f"\nCalculating embeddings\n")
    corpus_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), args.embedding_batch)):
            input_corpus = corpus[i:i+args.embedding_batch]
            batch_dict = retriever_tokenizer(input_corpus, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            corpus_embeddings.append(batch_embeddings.cpu())
    
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    
    query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), args.embedding_batch)):
            input_queries = queries[i:i+args.embedding_batch]
            batch_dict = retriever_tokenizer(input_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            query_embeddings.append(batch_embeddings.cpu())

    query_embeddings = torch.cat(query_embeddings, dim=0)

    bias_embeddings = []
    with torch.no_grad():
        input_bias = [args.projection_word]
        batch_dict = retriever_tokenizer(input_bias, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
        outputs = retriever_model(**batch_dict)
        batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        bias_embeddings.append(batch_embeddings.cpu())
    bias_embeddings = torch.cat(bias_embeddings, dim=1).reshape(-1, retriever_model.config.hidden_size)
    positive_embedding = bias_embeddings.reshape(-1, 1)

    B_T_A = torch.matmul(query_embeddings, positive_embedding)
    B_T_B = torch.matmul(positive_embedding.T, positive_embedding).item()
    positive_projection = torch.matmul(B_T_A/B_T_B, positive_embedding.T)
    orthogonal_pos = query_embeddings - positive_projection

    index_list = []
    for alpha in args.alpha:
        query_embeddings = orthogonal_pos + alpha * positive_projection
        index = get_index(query_embeddings, corpus_embeddings)
        index_list.append(index)

    return index_list

def get_index(query_embeddings, corpus_embeddings):
    norm_q_embeddings = normalize(query_embeddings, p=2, dim=1)
    norm_c_embeddings = normalize(corpus_embeddings, p=2, dim=1)
    scores = (norm_q_embeddings @ norm_c_embeddings.T)
    index = scores.topk(args.rag_type, dim=1)[1]

    return index

def calculate_index(queries, corpus):
    retriever_model, retriever_tokenizer = get_retriever(args.retriever, args)
    index_list = get_embeddings(retriever_model, retriever_tokenizer, queries, corpus)

    return index_list 

def save_results(bias_scores_list):
    results = {}
    for i, (bias1_scores_llm, bias2_scores_llm, bias1_scores_emb_llm, bias2_scores_emb_llm) in enumerate(bias_scores_list):
        bias1_llm = round(sum(bias1_scores_llm) / len(bias1_scores_llm) * 100, 1)
        bias2_llm = round(sum(bias2_scores_llm) / len(bias2_scores_llm) * 100, 1)
        bias1_emb_llm = round(sum(bias1_scores_emb_llm) / len(bias1_scores_emb_llm) * 100, 1)
        bias2_emb_llm = round(sum(bias2_scores_emb_llm) / len(bias2_scores_emb_llm) * 100, 1)
        bias_llm = bias1_llm - bias2_llm
        bias_emb_llm = bias1_emb_llm - bias2_emb_llm

        print(f"\nAlpha: {args.alpha[i]}\n")
        print(f"\nBias LLM Judge: {bias_llm} | Bias Emb LLM Judge: {bias_emb_llm}\n")

        results[args.alpha[i]] = {'bias1_llm': bias1_llm, 'bias2_llm': bias2_llm, 'bias1_emb_llm': bias1_emb_llm, 'bias2_emb_llm': bias2_emb_llm}

    results_path = f"{args.results_dir}/{args.llm.split('/')[-1]}-{args.query_dataset.split('/')[-1]}-{args.set}-{args.corpus_dataset.split('/')[-1]}-{args.projection_word}.json"
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
        existing_results.update(results)
        with open(results_path, 'w') as f:
            json.dump(existing_results, f)
    else:
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
    print("\nResults saved\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cache_dir", "--cache_dir", type=str, default=os.environ.get('HUG_CACHE_DIR'), help="Path to the cache directory")
    parser.add_argument("-results_dir", "--results_dir", type=str, default="results/project", help="Path to the results directory")
    parser.add_argument("-rm", "--retriever", type=str, default="thenlper/gte-base", help="Path to the retriever")
    parser.add_argument("-llm", "--llm", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", help="Path to the LLM") 
    parser.add_argument("-query_d", "--query_dataset", type=str, default='PoliticBias-QA', help="Dataset name for loading query model") 
    parser.add_argument("-corpus_d", "--corpus_dataset", type=str, default='Pol_NLI', help="Dataset name for loading corpus model") 
    parser.add_argument("-rag", "--rag_type", type=int, default=1, help="0: No RAG")
    parser.add_argument("-e_batch", "--embedding_batch", type=int, default=16, help="Batch size for embeddings")  
    parser.add_argument("-judge", "--judge", type=str, default="gpt-4o-mini", help="Path to the judge") 
    parser.add_argument("-set", "--set", type=str, default="test", help="Dataset split to use")
    parser.add_argument("-alpha", "--alpha", type=float, nargs='+', default=np.around(np.arange(1.0, 3.2, 0.2), decimals=1).tolist(), help="Alpha values for scaling") 
    parser.add_argument("-p_word", "--projection_word", type=str, default='republican', help="Word to project onto")
    parser.add_argument("-top_p", "--top_p", type=float, default=0.0, help="Top p value for generation")
    args = parser.parse_args()

    print(f"\nArguments: {args}\n")
    bias_queries, bias_corpus, bias_answers = get_bias_dataset(args.query_dataset, args.corpus_dataset, args.set, args.cache_dir)
    bias_index_list = calculate_index(bias_queries, bias_corpus)
    llm_model, llm_tokenizer = get_llm(args)
    bias_predictions_list = []
    for bias_index in bias_index_list:
        bias_predictions = get_predictions(llm_model, llm_tokenizer, bias_queries, bias_corpus, bias_index, bias_answers, args)
        bias_predictions_list.append(bias_predictions)
    del llm_model, llm_tokenizer
    
    topic_bias1, topic_bias2 = get_topics(args.query_dataset)
    bias_scores_list = []
    selected_corpus_list = []

    for bias_predictions, bias_index in zip(bias_predictions_list, bias_index_list):
        if args.query_dataset == 'PoliticBias-QA':
            bias1_scores_llm = [1 if x == topic_bias1[0] else 0 for x in bias_predictions]
            bias2_scores_llm = [1 if x == topic_bias2[0] else 0 for x in bias_predictions]
        else:
            bias1_scores_llm = get_bias_score(bias_predictions, topic=topic_bias1, judge=args.judge)
            bias2_scores_llm = get_bias_score(bias_predictions, topic=topic_bias2, judge=args.judge)

        selected_corpus = [bias_corpus[i] for i in bias_index]
        selected_corpus_list.append(selected_corpus)
        bias1_scores_emb_llm = get_bias_score(selected_corpus, topic=topic_bias1, judge=args.judge)
        bias2_scores_emb_llm = get_bias_score(selected_corpus, topic=topic_bias2, judge=args.judge)

        bias_scores_list.append((bias1_scores_llm, bias2_scores_llm, bias1_scores_emb_llm, bias2_scores_emb_llm))
    
        total_emb_bias = sum(bias1_scores_emb_llm)/len(bias1_scores_emb_llm) - sum(bias2_scores_emb_llm)/len(bias2_scores_emb_llm)
        total_bias = sum(bias1_scores_llm)/len(bias1_scores_llm) - sum(bias2_scores_llm)/len(bias2_scores_llm)
        print(f"\nTotal Bias: {total_bias}\nTotal Emb Bias: {total_emb_bias}\n")

    save_results(bias_scores_list)
    

    
    
    
    