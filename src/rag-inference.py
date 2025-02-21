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
from collections import Counter

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(retriever_model, retriever_tokenizer, queries, corpus, bias_answers):
    retriever_model.eval()
    print(f"\nCalculating embeddings\n")
    corpus_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), args.embedding_batch)):
            input_corpus = corpus[i:i+args.embedding_batch]

            if 'e5-base-v2' in args.retriever:
                input_corpus = [f'passage: {text}' for text in input_corpus]

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

            if 'e5-base-v2' in args.retriever:
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
    index = scores.topk(args.rag_type, dim=1)[1]

    return index

def calculate_index(queries, corpus, bias_answers, metric='acc'):
    index_path = os.path.join(args.index_dir, f'q-{args.query_dataset.split("/")[-1]}-{args.set}-c-{args.corpus_dataset.split("/")[-1]}-top-{args.rag_type}-{args.retriever.split("/")[-1]}.json')

    if os.path.exists(index_path) and metric == 'bias':
        print("\nIndices already exist\n")
        with open(index_path, 'r') as f:
            data = json.load(f)
        index = torch.tensor(data['index'])
    else:
        if args.rag_type == 0:
            print("\nNot using RAG\n")
            index = None
        elif isinstance(args.rag_type, int):
            print(f"\nUsing RAG Top-{args.rag_type}\n")   
            retriever_model, retriever_tokenizer = get_retriever(args.retriever, args)
            query_embeddings, corpus_embeddings = get_embeddings(retriever_model, retriever_tokenizer, queries, corpus, bias_answers)
            index = get_index(query_embeddings, corpus_embeddings)
        else:
            raise ValueError("Invalid RAG type")
        
        if index is not None and metric == 'bias':
            data = {'index': index.tolist()}
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            with open(index_path, 'w') as f:
                json.dump(data, f)
            
            print("\nIndices saved\n")
    return index

def save_results(bias1_scores, bias2_scores, bias1_scores_emb, bias2_scores_emb):

    bias1 = round(sum(bias1_scores) / len(bias1_scores) * 100, 1)
    bias2 = round(sum(bias2_scores) / len(bias2_scores) * 100, 1)
    if args.rag_type != 0:
        bias1_emb = round(sum(bias1_scores_emb) / len(bias1_scores_emb) * 100, 1)
        bias2_emb = round(sum(bias2_scores_emb) / len(bias2_scores_emb) * 100, 1)
    else:
        bias1_emb = None
        bias2_emb = None
    
    print(f"\nBias1: {bias1}, Bias2: {bias2}\n")    
    results = {'bias1': bias1, 'bias2': bias2, 'bias1_emb': bias1_emb, 'bias2_emb': bias2_emb, 'bias1_scores': bias1_scores, 'bias2_scores': bias2_scores}
    results_path = f"{args.results_dir}/{args.llm.split('/')[-1]}-{args.query_dataset.split('/')[-1]}-{args.set}-{args.corpus_dataset.split('/')[-1]}-topk-{args.rag_type}-{args.retriever.split('/')[-1]}.json"
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f)
        
    print("\nResults saved\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cache_dir", "--cache_dir", type=str, default=os.environ.get('HUG_CACHE_DIR'), help="Path to the cache directory")
    parser.add_argument("-index_dir", "--index_dir", type=str, default="dataset/infer_index", help="Path to the index directory")
    parser.add_argument("-results_dir", "--results_dir", type=str, default="results/inference", help="Path to the results directory")
    parser.add_argument("-rm", "--retriever", type=str, default="thenlper/gte-base", help="Path to the retriever") 
    parser.add_argument("-llm", "--llm", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to the LLM") 
    parser.add_argument("-query_d", "--query_dataset", type=str, default='PoliticBias-QA', help="Dataset name for loading query model") 
    parser.add_argument("-corpus_d", "--corpus_dataset", type=str, default='Pol_NLI', help="Dataset name for loading corpus model")
    parser.add_argument("-acc_d", "--accuracy_dataset", type=str, default='rag-datasets/mini_wikipedia', help="Dataset name for loading accuracy model")
    parser.add_argument("-rag", "--rag_type", type=int, default=1, help="0: No RAG")
    parser.add_argument("-e_batch", "--embedding_batch", type=int, default=16, help="Batch size for embeddings")  
    parser.add_argument("-judge", "--judge", type=str, default="gpt-4o-mini", help="Path to the judge") 
    parser.add_argument("-set", "--set", type=str, default="test", help="Dataset split to use")
    parser.add_argument("-top_p", "--top_p", type=float, default=0.0, help="Top p value for generation")
    args = parser.parse_args()
    
    print(f"\nArguments: {args}\n")

    bias_queries, bias_corpus, bias_answers = get_bias_dataset(args.query_dataset, args.corpus_dataset, args.set, args.cache_dir)

    bias_index = calculate_index(bias_queries, bias_corpus, bias_answers, 'bias')
    llm_model, llm_tokenizer = get_llm(args)
    bias_predictions = get_predictions(llm_model, llm_tokenizer, bias_queries, bias_corpus, bias_index, bias_answers, args)
    del llm_model, llm_tokenizer
    
    topic_bias1, topic_bias2 = get_topics(args.query_dataset)

    if args.query_dataset == 'PoliticBias-QA':
        bias1_scores = [1 if x == topic_bias1[0] else 0 for x in bias_predictions]
        bias2_scores = [1 if x == topic_bias2[0] else 0 for x in bias_predictions]
    else:
        bias1_scores = get_bias_score(bias_predictions, topic=topic_bias1, judge=args.judge)
        bias2_scores = get_bias_score(bias_predictions, topic=topic_bias2, judge=args.judge)

    if args.rag_type != 0:
        selected_corpus = [bias_corpus[i] for i in bias_index.flatten()]
        bias1_scores_emb = get_bias_score(selected_corpus, topic=topic_bias1, judge=args.judge)
        bias2_scores_emb = get_bias_score(selected_corpus, topic=topic_bias2, judge=args.judge)

        ### For the case of top-k > 1
        bias1_scores_emb = torch.tensor(bias1_scores_emb).reshape(-1, args.rag_type)
        bias2_scores_emb = torch.tensor(bias2_scores_emb).reshape(-1, args.rag_type)
        bias1_scores_emb = [Counter(bias1_scores_emb[i]).most_common(1)[0][0] for i in range(len(bias1_scores_emb))]
        bias1_scores_emb = torch.tensor(bias1_scores_emb).tolist()
        bias2_scores_emb = [Counter(bias2_scores_emb[i]).most_common(1)[0][0] for i in range(len(bias2_scores_emb))]
        bias2_scores_emb = torch.tensor(bias2_scores_emb).tolist()
    else:
        bias1_scores_emb = None
        bias2_scores_emb = None

    total_score = (sum(bias1_scores) - sum(bias2_scores)) / len(bias1_scores)
    print(f"\nTotal Score: {total_score}\n")
    
    if args.rag_type == 1:
        text_path = f"output_text"
        if not os.path.exists(text_path):
            os.makedirs(text_path)
        with open(f"{text_path}/{args.llm.split('/')[-1]}-{args.query_dataset.split('/')[-1]}-{args.set}-{args.corpus_dataset.split('/')[-1]}-topk-{args.rag_type}-{args.retriever.split('/')[-1]}.txt", 'w') as f:
            for i in range(len(bias_queries)):
                f.write(f"###{i}###\n")
                f.write(f"\nQuery: {bias_queries[i]}\n")
                f.write(f"\nDocument: {bias_corpus[bias_index[i][0]]}\n")
                f.write(f"\nPrediction: {bias_predictions[i]}\n")
                f.write(f"\n\nBias1 - Bias2: {bias1_scores[i]} - {bias2_scores[i]}\n")

    save_results(bias1_scores, bias2_scores, bias1_scores_emb, bias2_scores_emb)
    

    
    
    
    