import torch
import numpy as np
import random
import json
from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
from tqdm import tqdm
import csv

class DropLastDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        super(DropLastDataset, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = (len(dataset) // batch_size)
        self.length = self.n_batches * batch_size

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(f'Index {index} out of range for dataset of size {self.length}.')
        return self.dataset[index]

    def __len__(self):
        return self.length


class MixtureDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(MixtureDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum_lengths = np.cumsum(self.lengths)

    def __getitem__(self, index):
        dataset_index = np.searchsorted(self.cum_lengths, index, side='right')
        if dataset_index == 0:
            return self.datasets[0][index]
        else:
            return self.datasets[dataset_index][index-self.cum_lengths[dataset_index-1]]

    def __len__(self):
        return self.cum_lengths[-1]

class HomogenousBatchMixtureSampler(torch.utils.data.BatchSampler):
    def __init__(self, mixture_dataset, batch_size, generator=None, shuffle=True):
        if not isinstance(mixture_dataset, MixtureDataset):
            raise ValueError(f'MixtureDataset expected, got {type(mixture_dataset)}.')
        for dataset in mixture_dataset.datasets:
            if not isinstance(dataset, DropLastDataset):
                raise ValueError(f'DropLastDataset expected, instead got {type(dataset)}.')
            if batch_size != dataset.batch_size:
                raise ValueError(f'Batch size mismatch: {batch_size} != {dataset.batch_size}.')
        self.mixture_dataset = mixture_dataset
        self.batch_size = batch_size
        if generator is None:
            generator = torch.Generator()
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self):
        return len(self.mixture_dataset) // self.batch_size

    def __iter__(self):
        n_batches = len(self.mixture_dataset) // self.batch_size
        dataset_indices = torch.zeros(n_batches, dtype=torch.long)
        cur_batch = 0
        for i, dataset in enumerate(self.mixture_dataset.datasets):
            dataset_indices[cur_batch:cur_batch+dataset.n_batches] = i
            cur_batch += dataset.n_batches

        if self.shuffle:
            example_indices = [torch.randperm(len(d), generator=self.generator) for d in self.mixture_dataset.datasets]
            dataset_indices = dataset_indices[torch.randperm(len(dataset_indices), generator=self.generator)]
        else:
            example_indices = [torch.arange(len(d)) for d in self.mixture_dataset.datasets]

        current_batch_index = torch.zeros(len(self.mixture_dataset), dtype=torch.long)
        for i in range(len(self)):
            dataset_index = dataset_indices[i]
            example_index = example_indices[dataset_index][current_batch_index[dataset_index]:current_batch_index[dataset_index]+self.batch_size]
            current_batch_index[dataset_index] += self.batch_size
            offset = 0 if dataset_index == 0 else self.mixture_dataset.cum_lengths[dataset_index-1]
            yield example_index + offset

class JsonTaskMixtureDataset(MixtureDataset):
    def __init__(self, tasks, loss_type, dataset_dir=None, bias_set=None, index_dir_list=None, cache_dir=None, drop_last_each=False, batch_size=None, shuffle_dataset_order=False, transform=None):
        if dataset_dir is None:
            raise ValueError('dataset_dir must be specified.')
        data = []   
        
        dataset = load_from_disk(dataset_dir)
        dataset = dataset[bias_set]
        queries = dataset["queries"]
        corpus_index_list = []
        print(f"Index directories: {index_dir_list}")
        for index_dir in index_dir_list:
            with open(index_dir, 'r') as file:
                index_dict = json.load(file)
            
            pos_index = index_dict["pos_docs"]
            neutral_index = index_dict["neutral_docs"]
            neg_index = index_dict["neg_docs"]
            dataset_name = index_dict["queries"]  

            if 'nq' in index_dir:
                corpus_dataset = load_dataset('mteb/nq', "corpus", cache_dir=cache_dir)
                corpus = corpus_dataset['corpus']['text']
            elif 'msmarco' in index_dir:
                corpus_dataset = load_dataset('mteb/msmarco', "corpus", cache_dir=cache_dir)
                corpus = corpus_dataset['corpus']['text']
            elif 'hotpotqa' in index_dir:
                corpus_dataset = load_dataset('mteb/hotpotqa', "corpus", cache_dir=cache_dir)
                corpus = corpus_dataset['corpus']['text']
            elif 'fever' in index_dir:
                corpus_dataset = load_dataset('mteb/fever', "corpus", cache_dir=cache_dir)
                corpus = corpus_dataset['corpus']['text']
            elif 'dbpedia' in index_dir:
                corpus_dataset = load_dataset('mteb/dbpedia', "corpus", cache_dir=cache_dir)
                corpus = corpus_dataset['corpus']['text']
            elif 'args_me' in index_dir:
                corpus_dataset = load_dataset('webis/args_me', 'corpus', cache_dir=cache_dir, streaming=True, trust_remote_code=True)
                corpus_dataset = iter(corpus_dataset['train'])
                corpus = []
                for datapoint in corpus_dataset:
                    corpus.append(datapoint['argument'])
                corpus = [text for text in corpus if '?' not in text]
            elif 'webis-argument-framing' in index_dir:
                with open('dataset/tasks/webis-argument-framing.csv', mode='r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    corpus = [row[5] for row in csv_reader][1:]
                corpus = [text for text in corpus if '?' not in text]
            elif 'conclugen' in index_dir:
                corpus = load_dataset("webis/conclugen", "aspects", cache_dir=cache_dir)
                corpus = corpus['train']
                corpus = corpus['conclusion']
                corpus = [text for text in corpus if '?' not in text]

            corpus_index_list.append((pos_index, neutral_index, neg_index, corpus))
        
        if loss_type == 'simcse':
            print(f"\nSorting documents for training")
            for num, query in tqdm(enumerate(queries)):
                pos_docs = []
                neutral_docs = []
                neg_docs = []
                for (pos_index, neutral_index, neg_index, corpus) in corpus_index_list:
                    pos_docs += [corpus[i] for i in pos_index[num]]
                    neutral_docs += [corpus[i] for i in neutral_index[num]]
                    neg_docs += [corpus[i] for i in neg_index[num]]
                pos_docs = list(set(pos_docs))
                neutral_docs = list(set(neutral_docs))
                neg_docs = list(set(neg_docs))

                limit_num = 10
                if len(pos_docs) >= limit_num:
                    positive = random.sample(pos_docs, limit_num)
                elif len(pos_docs) > 0:
                    positive = pos_docs
                else:
                    continue

                if 'PoliticBias-QA' in dataset_name:
                    if len(neg_docs) >= limit_num:
                        negative = random.sample(neg_docs, limit_num)
                    elif len(neg_docs) > 0:
                        negative = neg_docs
                    else:
                        continue
                else:
                    if len(neg_docs) >= limit_num:
                        negative = random.sample(neg_docs, limit_num)
                    elif len(neg_docs + neutral_docs) >= limit_num:
                        negative = random.sample(neg_docs + neutral_docs, limit_num)
                    elif len(neg_docs + neutral_docs) > 0:
                        negative = neg_docs + neutral_docs
                    else:
                        continue

                entry = {
                    'query': query,
                    'positive': positive,
                    'negative': negative, 
                }

                data.append(self.json_to_entry(entry, dataset_name))
            
            del corpus_index_list
        else:
            raise ValueError(f'Invalid loss type: {loss_type}')
         
        self.data = data
        self.task_ids = set([x['task'] for x in data])
        self.mixtures = [[x for x in data if x['task'] == task_id] for task_id in self.task_ids]
        if drop_last_each:
            if batch_size is None:
                raise ValueError('batch_size must be specified if drop_last_each is True.')
            self.mixtures = [DropLastDataset(m, batch_size=batch_size) for m in self.mixtures]
        if shuffle_dataset_order:
            random.shuffle(self.mixtures)
        self.transform = transform
        super(JsonTaskMixtureDataset, self).__init__(self.mixtures)

    @staticmethod
    def json_to_entry(obj, task):
        return {
            'task': task,
            'query': obj['query'],
            'pos': obj['positive'],
            'neg': obj['negative'],
        }
        
    @staticmethod
    def json_to_entry_contrast(obj, task):
        return {
            'task': task,
            'query': obj['query'],
            'pos': obj['positive'],
            'neg': obj['negative'],
            'cont_neg': obj['contrastive']
        }

    def __getitem__(self, index):
        data = super(JsonTaskMixtureDataset, self).__getitem__(index)
        if self.transform is not None:
            data = self.transform(data)
        return data