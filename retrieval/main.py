''' retrieval of vanilla rag baseline '''

from retrieval import (
    CorpusLoader,
    QueriesLoader,
    DenseRetrievalExactSearch as DR,
    DenseRetrievalExactSearchMultiDatasets as DRMD,
)
from retrievers import (
    BGE,
    DPR,
    Contriever,
)
from typing import List
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def passages_retrieval(dataset: str,
                       corpus_file: str,
                       base_folder: str = '',
                       seed_id: str = '0',
                       model_name: str = 'dpr',
                       top_k: int = 5,
                       batch_size: int = 128,
                       task_instruction: str='Given a web search query, retrieve relevant passages that answer the query'
                       corpus_chunk_size: int = 100000):
    '''
    retrieve relevant passages for shared context construction based on entities
    '''
    filename = os.path.join(base_folder, dataset, f'test_{seed_id}.jsonl')
    queries, inputs, outputs = QueriesLoader(
        data_path=filename, query_type='input', task_instruction=task_instruction).load()
    corpus = CorpusLoader(corpus_path=corpus_file).load()

    if model_name == 'dpr':
        model = DPR((
            'facebook/dpr-question_encoder-multiset-base',
            'facebook/dpr-ctx_encoder-multiset-base'
        )),
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-base' or model_name == 'bge':
        model = BGE('BAAI//bge-base-en-v1.5')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    else:
        logging.error(f'Wrong retriever: {model_name}')

    results = retriever.search(corpus, queries, top_k=top_k, score_function='cos_sim', return_sorted=False)

    # dict_keys(['_id', 'input', 'docs', 'output'])
    output_filename = os.path.join(base_folder, dataset, f'test_{seed_id}_w_passages_{model_name}.jsonl')    
    with open(output_filename, 'w') as file:
        for query_id, ranking_scores in results.items():
            sorted_scores = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
            docs = [{'title': corpus[doc_id].get('title'), 'text': corpus[doc_id].get('text'), 'score': score} for doc_id, score in sorted_scores]
            line = {'_id': query_id, 'input': inputs[query_id], 'docs': docs, 'output': outputs[query_id]}
            file.write(json.dumps(line) + '\n')

    return {
        'output_file': output_filename
    }


def main():
    wiki_datasets = ['nq', 'hotpotqa', 'eli5', 'fever', 'wow', 'trex', 'zs-re']
    beir_datasets = ['fiqa', 'climate-fever', 'scifact']
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', required=True, type=str, help='The retriever model path.')
    parser.add_argument('--corpus_path', required=True, type=str, help='The corpus file path.')
    parser.add_argument('--topk', default=5, required=True, type=int)
    args = parser.parse_args()

    if 'wiki' in args.corpus_path:
        outputs = passages_retrieval_multiset(
            dataset_list=wiki_datasets,
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'fiqa' in args.corpus_path:
        outputs = passages_retrieval(
            dataset='fiqa',
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'climate' in args.corpus_path:
        outputs = passages_retrieval(
            dataset='climate-fever',
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'scifact' in args.corpus_path:
        outputs = passages_retrieval(
            dataset='scifact',
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'pooled' in args.corpus_path:
        outputs = passages_retrieval_multiset(
            dataset_list=wiki_datasets + beir_datasets,
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )


if __name__ == '__main__':
    main()
