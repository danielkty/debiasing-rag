import os
import torch.nn as nn
import json
from transformers import AutoModel, AutoTokenizer

import embedder.templates as templates
from embedder.parser import TemplateParser, TemplatePooling
    
class GteEmbedder(nn.Module):
    def __init__(self, model_path, template, pooling, max_length, cache_dir):
        super(GteEmbedder, self).__init__()
        self.template = template
        self.pooling = pooling
        self.max_length = max_length
        self.parser = TemplateParser(
            tokenizer_path=model_path,
            templates=templates.GTE_TEMPLATES[template],
            max_length=max_length,
        )
        self.model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
        self.pooler = TemplatePooling(pooling)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, xs):
        xs = self.parser(xs)
        inputs = {
            'input_ids': xs['input_ids'].to(self.model.device),
            'attention_mask': xs['attention_mask'].to(self.model.device),
        }
        outputs = self.model(**inputs).last_hidden_state
        xs['token_embeddings'] = outputs
        xs = self.pooler(xs)
        return xs

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.parser.tokenizer.save_pretrained(path)
        with open(path + '/embedder_config.json', 'w') as f:
            json.dump({
                'template': self.template, 
                'pooling': self.pooling, 
                'max_length': self.max_length}, 
                f, indent=4)

    @staticmethod
    def from_pretrained(model_path, template=None, pooling=None, max_length=None):
        if os.path.exists(model_path + '/embedder_config.json'):
            with open(model_path + '/embedder_config.json', 'r') as f:
                config = json.load(f)
            if template is None:
                template = config['template']
            if pooling is None:
                pooling = config['pooling']
            if max_length is None:
                max_length = config['max_length']
        if template is None:
            raise ValueError('template must be specified if loading from a pretrained model')
        if pooling is None:
            raise ValueError('pooling must be specified if loading from a pretrained model')
        if max_length is None:
            raise ValueError('max_length must be specified if loading from a pretrained model')
        return GteEmbedder(model_path, template, pooling, max_length)