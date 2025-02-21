from datasets import load_dataset, load_from_disk
import csv

def get_bias_dataset(query_dataset, corpus_dataset, split, cache_dir):
    if query_dataset == 'PoliticBias-QA':
        file_path = f'dataset/tasks/{query_dataset}'
        dataset = load_from_disk(file_path)
        dataset = dataset[split]
        queries = dataset["queries"]
        answers = [dataset["left_claims"], dataset["right_claims"]]

    elif query_dataset == 'GenderBias-QA':
        file_path = f'dataset/tasks/{query_dataset}'
        dataset = load_from_disk(file_path)
        dataset = dataset[split]
        queries = dataset["queries"]
    else:
        raise ValueError("Unknown query dataset. Implement proper loading function.")

    if corpus_dataset == 'PoliticBias-QA':
        file_path = f'dataset/tasks/{corpus_dataset}'
        dataset = load_from_disk(file_path)
        dataset = dataset["corpus"]
        corpus = dataset["test"]
    elif corpus_dataset == 'GenderBias-QA':
        file_path = f'dataset/tasks/{corpus_dataset}'
        dataset = load_from_disk(file_path)
        dataset = dataset["corpus"]
        corpus = dataset["text"]
    elif 'mteb' in corpus_dataset:
        corpus_dataset = load_dataset(corpus_dataset, "corpus", cache_dir=cache_dir)
        corpus = corpus_dataset['corpus']['text']
    elif corpus_dataset == 'webis/args_me':
        data = load_dataset('webis/args_me', 'corpus', cache_dir=cache_dir, streaming=True, trust_remote_code=True)
        data = iter(data['train'])
        corpus = []
        for datapoint in data:
            corpus.append(datapoint['argument'])
        corpus = [text for text in corpus if '?' not in text]
    elif corpus_dataset == 'webis-argument-framing':
        with open('dataset/tasks/webis-argument-framing.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            corpus = [row[5] for row in csv_reader][1:]
        corpus = [text for text in corpus if '?' not in text]
    elif corpus_dataset == 'webis/conclugen':
        data = load_dataset("webis/conclugen", "aspects", cache_dir=cache_dir)
        corpus = data['train']
        corpus = corpus['conclusion']
        corpus = [text for text in corpus if '?' not in text]
    elif 'Pol_NLI' in corpus_dataset:
        data = load_dataset("mlburnham/Pol_NLI", cache_dir=cache_dir)
        data = data['train']
        corpus = data['premise']
    else:
        raise ValueError("Unknown corpus dataset. Implement proper loading function.")

    if 'answers' not in locals():
        answers = None
        
    return queries, corpus, answers