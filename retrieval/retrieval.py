''' passages retrieval for context construction (reference from repository @beir) '''

from tqdm import tqdm
from typing import Dict, List
import os
import json
import torch
import heapq
import logging
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CorpusLoader:

    def __init__(self, corpus_path: str = None):
        self.corpus = {}
        self.corpus_file = corpus_path
    
    @staticmethod
    def check(in_file: str, ext: str):
        if not os.path.exists(in_file):
            raise ValueError('File {} not present! Please provide accurate file.'.format(in_file))
        if not in_file.endswith(ext):
            raise ValueError('File {} must be present with extension {}'.format(in_file, ext))
    
    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as in_file:
            for line in tqdm(in_file, total=num_lines):
                line = json.loads(line)
                # the format is consistent with Wikipedia and Beir
                self.corpus[line.get('_id')] = {
                    'text': line.get('text'),
                    'title': line.get('title'),
                }
    
    def load(self) -> Dict[str, Dict[str, str]]:
        self.check(in_file=self.corpus_file, ext='jsonl')
        # only needs to load on first access
        if not len(self.corpus):
            logger.info('Loading Corpus...')
            self._load_corpus()
            logger.info('Loaded %d Documents.', len(self.corpus))

        return self.corpus


class QueriesLoader:

    def __init__(self,
                 data_path: str = None,
                 query_type: str = 'entity',
                 task_instruction: str = None):
        '''
        - data_path: the jsonl file's keys can be
            - dict_keys(['_id', 'entities'])
            - dict_keys(['_id', 'input', 'output'])
        - query_type:
            - 'entity': construct query with entities
            - 'input': construct query with input
        '''
        self.queries = {}
        self.data_file = data_path
        self.query_type = query_type
        if self.query_type == 'entity':
            self.mapping = {}
        elif self.query_type == 'input':
            self.instruction = task_instruction
            self.inputs = {}
            self.outputs = {}
    
    @staticmethod
    def check(in_file: str, ext: str):
        if not os.path.exists(in_file):
            raise ValueError('File {} not present! Please provide accurate file.'.format(in_file))
        if not in_file.endswith(ext):
            raise ValueError('File {} must be present with extension {}'.format(in_file, ext))
    
    def _load_queries(self):
        with open(self.data_file, encoding='utf8') as in_file:
            for line in in_file:
                line = json.loads(line)
                if self.query_type == 'entity':
                    self.mapping[line.get('_id')] = []
                    for idx, entity in enumerate(line.get('entities')):
                        self.queries[line.get('_id') + str(idx)] = entity
                        self.mapping[line.get('_id')].append(line.get('_id') + str(idx))
                    # self.queries[line.get('_id')] = self._generate_entity_query(line.get('entities'))
                elif self.query_type == 'input':
                    if self.instruction is None:
                        self.queries[line.get('_id')] = line.get('input')
                    else:
                        self.queries[line.get('_id')] = self.instruction + '[SEP]' + line.get('input')
                    self.inputs[line.get('_id')] = line.get('input')
                    self.outputs[line.get('_id')] = line.get('output')
                else:
                    raise ValueError(f'no type calls {self.query_type}')
    
    def load(self) -> Dict[str, str]:
        self.check(in_file=self.data_file, ext='jsonl')
        # only needs to load on first access
        if not len(self.queries):
            logger.info('Loading Queries...')
            self._load_queries()
            logger.info('Loaded %d Queries.', len(self.queries))
        
        if self.query_type == 'entity':
            return self.queries, self.mapping
        elif self.query_type == 'input':
            return self.queries, self.inputs, self.outputs


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    '''
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    '''
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1)) #TODO: this keeps allocating GPU memory


def dot_score(a: torch.Tensor, b: torch.Tensor):
    '''
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    '''
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


class DenseRetrievalExactSearch:
    
    def __init__(self, model, batch_size: int = 512, corpus_chunk_size: int = 10000, **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': 'Cosine Similarity', 'dot': 'Dot Product'}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get('show_progress_bar', True)
        self.convert_to_tensor = kwargs.get('convert_to_tensor', True)
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:

        if score_function not in self.score_functions:
            raise ValueError('score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product'.format(score_function))
            
        logger.info('Encoding Queries...')
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
          
        logger.info('Sorting Corpus by document length (Longest first)...')

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get('title', '') + corpus[k].get('text', '')), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids] # [{'text': ..., 'title': ...}...]

        logger.info('Encoding Corpus in batches... Warning: This might take a while!')
        logger.info('Scoring Function: {} ({})'.format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info('Encoding Batch {}/{}...'.format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx: corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted) # torch.Size([97852, 21]) <- [num_queries, top_k + 1]
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
            

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        
        return self.results 


class DenseRetrievalExactSearchMultiDatasets:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 10000, **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': 'Cosine Similarity', 'dot': 'Dot Product'}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = []
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries_list: List[Dict[str, str]],
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        
        if score_function not in self.score_functions:
            raise ValueError('score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product'.format(score_function))
        
        logger.info('Encoding Queries...')

        query_embeddings_list = []
        query_ids_list = []

        for queries in queries_list:
            query_ids = list(queries.keys())
            query_ids_list.append(query_ids)
            self.results.append({qid: {} for qid in query_ids})
            queries = [queries[qid] for qid in queries]
            query_embeddings = self.model.encode_queries(queries,
                                                         batch_size=self.batch_size,
                                                         show_progress_bar=self.show_progress_bar,
                                                         convert_to_tensor=self.convert_to_tensor)
            query_embeddings_list.append(query_embeddings)
        
        logger.info('Sorting Corpus by document length (Longest first)...')

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get('title', '') + corpus[k].get('text', '')), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids] # [{'text': ..., 'title': ...}...]

        logger.info('Encoding Corpus in batches... Warning: This might take a while!')
        logger.info('Scoring Function: {} ({})'.format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps_list = [{qid: [] for qid in query_ids} for query_ids in query_ids_list] # Keep only the top-k docs for each query

        for batch_num, corpus_start_idx in enumerate(itr):
            # if batch_num == 0:
            #     continue
            logger.info('Encoding Batch {}/{}...'.format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx: corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )
            
            for query_embeddings, query_ids, result_heaps in zip(query_embeddings_list, query_ids_list, result_heaps_list):

                # Compute similarites using either cosine-similarity or dot product
                cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
                cos_scores[torch.isnan(cos_scores)] = -1
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
                for query_itr in range(len(query_embeddings)):
                    query_id = query_ids[query_itr]
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                        if corpus_id != query_id:
                            if len(result_heaps[query_id]) < top_k:
                                # Push item on the heap
                                heapq.heappush(result_heaps[query_id], (score, corpus_id))
                            else:
                                # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                                heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
            

        for idx, result_heaps in enumerate(result_heaps_list):
            for qid in result_heaps:
                for score, corpus_id in result_heaps[qid]:
                    self.results[idx][qid][corpus_id] = score
        
        return self.results 

