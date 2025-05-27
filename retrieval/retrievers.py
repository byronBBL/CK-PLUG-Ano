from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    AutoTokenizer,
    AutoModel,
)
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch

class DPR:
    def __init__(self,
                 model_path: Union[str, Tuple] = None,
                 **kwargs):
        # Query tokenizer and model
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0])
        # self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0], device_map='auto', torch_dtype=torch.float16)
        self.q_model.cuda()
        self.q_model.eval()
        
        # Context tokenizer and model
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1])
        # self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1], device_map='auto', torch_dtype=torch.float16)
        self.ctx_model.cuda()
        self.ctx_model.eval()
    
    def encode_queries(self,
                       queries: List[str],
                       batch_size: int = 16,
                       **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = self.q_tokenizer(queries[start_idx: start_idx + batch_size], truncation=True, padding=True, return_tensors='pt', max_length=512)
                model_out = self.q_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                # model_out = self.q_model(encoded['input_ids'].to(self.q_model.device), attention_mask=encoded['attention_mask'].to(self.q_model.device))
                query_embeddings += model_out.pooler_output

        return torch.stack(query_embeddings)
        
    def encode_corpus(self,
                      corpus: List[Dict[str, str]],
                      batch_size: int = 8,
                      **kwargs) -> torch.Tensor:
        
        corpus_embeddings = []
        with torch.no_grad():
            # for start_idx in trange(0, len(corpus), batch_size):
            for start_idx in range(0, len(corpus), batch_size):
                titles = [row['title'] for row in corpus[start_idx: start_idx + batch_size]]
                texts = [row['text']  for row in corpus[start_idx: start_idx + batch_size]]
                encoded = self.ctx_tokenizer(titles, texts, max_length=512, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.ctx_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                # model_out = self.ctx_model(encoded['input_ids'].to(self.ctx_model.device), attention_mask=encoded['attention_mask'].to(self.ctx_model.device))
                corpus_embeddings += model_out.pooler_output.detach()
        
        return torch.stack(corpus_embeddings)


class BGE:
    def __init__(self,
                 model_path: str = None,
                 **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.cuda()
        self.model.eval()
    
    def encode_queries(self,
                       queries: List[str],
                       batch_size: int = 16,
                       normalize: bool = True,
                       **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = self.tokenizer(queries[start_idx: start_idx + batch_size], truncation=True, padding=True, return_tensors='pt')
                model_out = self.model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                # model_out = self.model(encoded['input_ids'].to(self.q_model.device), attention_mask=encoded['attention_mask'].to(self.q_model.device))
                query_embedding = model_out[0][:, 0]
                if normalize:
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                query_embeddings.append(query_embedding)

        return torch.cat(query_embeddings, dim=0)
    
    def encode_corpus(self,
                      corpus: List[Dict[str, str]],
                      batch_size: int = 8,
                      normalize: bool = True,
                      **kwargs) -> torch.Tensor:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(corpus), batch_size):
                docs = [row['title'] + '\n' + row['text'] for row in corpus[start_idx: start_idx + batch_size]]
                encoded = self.tokenizer(docs, max_length=512, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                # model_out = self.ctx_model(encoded['input_ids'].to(self.ctx_model.device), attention_mask=encoded['attention_mask'].to(self.ctx_model.device))
                corpus_embedding = model_out[0][:, 0]
                if normalize:
                    corpus_embedding = torch.nn.functional.normalize(corpus_embedding, p=2, dim=1)
                corpus_embeddings.append(corpus_embedding)
        
        return torch.cat(corpus_embeddings, dim=0)


class Contriever:
    def __init__(self,
                 model_path: str = None,
                 **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.cuda()
        self.model.eval()

    def encode_queries(self,
                       queries: List[str],
                       batch_size: int = 16,
                       normalize: bool = True,
                       **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = self.tokenizer(queries[start_idx: start_idx + batch_size], truncation=True, padding=True, return_tensors='pt')
                model_out = self.model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                # model_out = self.model(encoded['input_ids'].to(self.q_model.device), attention_mask=encoded['attention_mask'].to(self.q_model.device))
                query_embedding = model_out[0][:, 0]
                if normalize:
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                query_embeddings.append(query_embedding)

        return torch.cat(query_embeddings, dim=0)
    
    def encode_corpus(self,
                      corpus: List[Dict[str, str]],
                      batch_size: int = 8,
                      normalize: bool = True,
                      **kwargs) -> torch.Tensor:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(corpus), batch_size):
                docs = [row['title'] + '\n' + row['text'] for row in corpus[start_idx: start_idx + batch_size]]
                encoded = self.tokenizer(docs, max_length=512, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                # model_out = self.ctx_model(encoded['input_ids'].to(self.ctx_model.device), attention_mask=encoded['attention_mask'].to(self.ctx_model.device))
                corpus_embedding = model_out[0][:, 0]
                if normalize:
                    corpus_embedding = torch.nn.functional.normalize(corpus_embedding, p=2, dim=1)
                corpus_embeddings.append(corpus_embedding)
        
        return torch.cat(corpus_embeddings, dim=0)

