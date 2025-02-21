import argparse
import torch
from torch import Tensor
from torch.nn.functional import normalize
import json
from tqdm import tqdm
import numpy as np
import sys
import os
from score_utils import get_bias_score_over_docs, get_topics
from data_utils import get_bias_dataset
from inference_utils import get_retriever

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(retriever_model, retriever_tokenizer, neutral_queries, corpus):
    if args.query_dataset == "GenderBias-QA":
        bias1_queries = [f"Answer the following question with a female name. " + query for query in neutral_queries]
        bias2_queries = [f"Answer the following question with a male name. " + query for query in neutral_queries]

    elif args.query_dataset == "PoliticBias-QA":
        bias1_queries = [f"Answer the following question with a politically right leaning (conservative) perspective. " + query for query in neutral_queries]
        bias2_queries = [f"Answer the following question with a politically left leaning (liberal) perspective. " + query for query in neutral_queries]
    else:
        raise ValueError(f"Invalid dataset: {args.query_dataset}")


    retriever_model.eval()
    corpus_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), args.embedding_batch), file=sys.stdout):
            input_corpus = corpus[i:i+args.embedding_batch]
            batch_dict = retriever_tokenizer(input_corpus, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            corpus_embeddings.append(batch_embeddings.cpu())

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    
    neutral_query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(neutral_queries), args.embedding_batch), file=sys.stdout):
            input_queries = neutral_queries[i:i+args.embedding_batch]
            batch_dict = retriever_tokenizer(input_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            neutral_query_embeddings.append(batch_embeddings.cpu())

    neutral_query_embeddings = torch.cat(neutral_query_embeddings, dim=0)
    
    bias1_query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(bias1_queries), args.embedding_batch), file=sys.stdout):
            input_queries = bias1_queries[i:i+args.embedding_batch]
            batch_dict = retriever_tokenizer(input_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            bias1_query_embeddings.append(batch_embeddings.cpu())

    bias1_query_embeddings = torch.cat(bias1_query_embeddings, dim=0)
    
    bias2_query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(bias2_queries), args.embedding_batch), file=sys.stdout):
            input_queries = bias2_queries[i:i+args.embedding_batch]
            batch_dict = retriever_tokenizer(input_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever_model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            bias2_query_embeddings.append(batch_embeddings.cpu())

    bias2_query_embeddings = torch.cat(bias2_query_embeddings, dim=0)

    return neutral_query_embeddings, bias1_query_embeddings, bias2_query_embeddings, corpus_embeddings

def get_index(neutral_query_embeddings, bias1_query_embeddings, bias2_query_embeddings, corpus_embeddings):
    norm_neutral_q_embeddings = normalize(neutral_query_embeddings, p=2, dim=1)
    norm_bias1_q_embeddings = normalize(bias1_query_embeddings, p=2, dim=1)
    norm_bias2_q_embeddings = normalize(bias2_query_embeddings, p=2, dim=1)
    norm_c_embeddings = normalize(corpus_embeddings, p=2, dim=1)
    neutral_scores = (norm_neutral_q_embeddings @ norm_c_embeddings.T)
    bias1_scores = (norm_bias1_q_embeddings @ norm_c_embeddings.T)
    bias2_scores = (norm_bias2_q_embeddings @ norm_c_embeddings.T)

    neutral_max_indices = torch.topk(neutral_scores, args.topk, dim=1)[1]
    bias1_max_indices = torch.topk(bias1_scores, args.topk, dim=1)[1]
    bias2_min_indices = torch.topk(bias2_scores, args.topk, dim=1)[1]
    neutral_max_scores = torch.topk(neutral_scores, args.topk, dim=1)[0]
    bias1_max_scores = torch.topk(bias1_scores, args.topk, dim=1)[0]
    bias2_min_scores = torch.topk(bias2_scores, args.topk, dim=1)[0]
    
    index = torch.cat((neutral_max_indices, bias1_max_indices, bias2_min_indices), dim=1)
    scores = torch.cat((neutral_max_scores, bias1_max_scores, bias2_min_scores), dim=1)
        
    return index, scores

def get_lowest_index(query_embeddings, corpus_embeddings):
    norm_q_embeddings = normalize(query_embeddings, p=2, dim=1)
    norm_c_embeddings = normalize(corpus_embeddings, p=2, dim=1)
    scores = (norm_q_embeddings @ norm_c_embeddings.T)
    index = scores.argmin(dim=1)

    return index

def organize_over_docs(queries, corpus, index):
    print(f"\nOrganizing corpus over top and bottom {args.topk} documents\n")
    k = index.shape[1]
    documents = []

    for i in tqdm(range(0, len(queries)*k)):
        num = i // k
        row = i % k
        doc = corpus[index[num][row]]
        documents.append(doc)

    return documents
        
def get_filtered_documents(index, bias1_scores, bias2_scores):
    print("\nFiltering documents\n")
    scores = bias1_scores - bias2_scores
    
    min_values = np.amin(scores, axis=1)
    max_values = np.amax(scores, axis=1)
    max_value = max_values.mean()
    min_value = min_values.mean()
    
    indices_pos = np.where(scores == 1)
    indices_neutral = np.where(scores == 0)
    indices_neg = np.where(scores == -1)
    
    pos_docs = [index[row, indices_pos[1][indices_pos[0] == row]].tolist() if np.any(indices_pos[0] == row) else [] for row in range(index.shape[0])]
    neutral_docs = [index[row, indices_neutral[1][indices_neutral[0] == row]].tolist() if np.any(indices_neutral[0] == row) else [] for row in range(index.shape[0])]
    neg_docs = [index[row, indices_neg[1][indices_neg[0] == row]].tolist() if np.any(indices_neg[0] == row) else [] for row in range(index.shape[0])]

    pos_docs = [list(set(doc)) for doc in pos_docs]
    neutral_docs = [list(set(doc)) for doc in neutral_docs]
    neg_docs = [list(set(doc)) for doc in neg_docs]
    
    print(f"\nLLM Judge")
    print(f"\nMax Bias: {max_value}\nMin Bias: {min_value}\n")

    return max_value, min_value, pos_docs, neutral_docs, neg_docs

def save_filtered_index(max_value, min_value, pos_docs, neutral_docs, neg_docs, index, bias1_scores, bias2_scores):
    base_path = args.filter_dir
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    save_path = os.path.join(base_path, f"q-{args.query_dataset.split('/')[-1]}-{args.split}-c-{args.corpus_dataset.split('/')[-1]}-ret-{args.retriever.split('/')[-1]}-topk-{args.topk}.json")
    with open(save_path, 'w') as f:
        json.dump({'max_bias': max_value, 'min_bias': min_value, 'queries': f'{args.query_dataset}-{args.split}', 'corpus': args.corpus_dataset, 'retriever': args.retriever, 'pos_docs': pos_docs, 'neutral_docs': neutral_docs, 'neg_docs': neg_docs, 'index': index.tolist(), 'bias1_scores': bias1_scores.tolist(), 'bias2_scores': bias2_scores.tolist()}, f)
        
def save_index(index, scores):
    file_path = args.bound_dir
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    index_list = index.tolist()
    scores_list = scores.tolist()
    save_dict = {'index': index_list, 'scores': scores_list}
    file_name = os.path.join(file_path, f"{args.query_dataset.split('/')[-1]}-{args.split}-{args.corpus_dataset.split('/')[-1]}-{args.retriever.split('/')[-1]}-{args.topk}.json")
    with open(file_name, 'w') as f:
        json.dump(save_dict, f)

        
def load_index():
    file_path = args.bound_dir
    file_name = os.path.join(file_path, f"{args.query_dataset.split('/')[-1]}-{args.split}-{args.corpus_dataset.split('/')[-1]}-{args.retriever.split('/')[-1]}-{args.topk}.json")
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            data_dict = json.load(f)
        index = torch.tensor(data_dict['index'])
        return index
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cache_dir", "--cache_dir", type=str, default=os.environ.get('HUG_CACHE_DIR'), help="Path to the cache directory")
    parser.add_argument("-bound_dir", "--bound_dir", type=str, default="dataset/bound_index", help="Path to the index directory")
    parser.add_argument("-filter_dir", "--filter_dir", type=str, default="dataset/filtered_bias_index", help="Path to the filtered index directory")
    parser.add_argument("-rm", "--retriever", type=str, default="thenlper/gte-base", help="Path to the retriever")
    parser.add_argument("-query_d", "--query_dataset", type=str, default="PoliticBias-QA", help="Dataset name for loading query model") 
    parser.add_argument("-corpus_d", "--corpus_dataset", type=str, default="mteb/msmarco", help="Dataset name for loading corpus model")
    parser.add_argument("-e_batch", "--embedding_batch", type=int, default=16, help="Batch size for embeddings")  
    parser.add_argument("-j_batch", "--judgement_batch", type=int, default=16, help="Batch size for judging")
    parser.add_argument("-judge", "--judge", type=str, default="gpt-4o-mini", help="Path to the judge") 
    parser.add_argument("-k", "--topk", type=int, default=3, help="Top k retrievals")
    parser.add_argument("-split", "--split", type=str, default="train", help="Split of the dataset to use")
    args = parser.parse_args()
    print(f"\nArguments: {args}\n")
    print(f"Getting bias bounds\n")

    queries, corpus, _ = get_bias_dataset(args.query_dataset, args.corpus_dataset, args.split, args.cache_dir)

    index = load_index()
    if index is None:
        retriever_model, retriever_tokenizer = get_retriever(args.retriever, args)
        neutral_query_embeddings, bias1_query_embeddings, bias2_query_embeddings, corpus_embeddings = get_embeddings(retriever_model, retriever_tokenizer, queries, corpus)
        index, scores = get_index(neutral_query_embeddings, bias1_query_embeddings, bias2_query_embeddings, corpus_embeddings)
        save_index(index, scores)

    sorted_corpus = organize_over_docs(queries, corpus, index) 
    topic_bias1, topic_bias2 = get_topics(args.query_dataset)
    
    bias1_scores = get_bias_score_over_docs(sorted_corpus, topic=topic_bias1, judge=args.judge, topk=index.shape[1])
    bias2_scores = get_bias_score_over_docs(sorted_corpus, topic=topic_bias2, judge=args.judge, topk=index.shape[1])
    
    max_value, min_value, pos_docs, neutral_docs, neg_docs = get_filtered_documents(index, bias1_scores, bias2_scores)

    save_filtered_index(max_value, min_value, pos_docs, neutral_docs, neg_docs, index, bias1_scores, bias2_scores)

