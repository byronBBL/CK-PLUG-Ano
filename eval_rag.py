import argparse
import json
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from eval_rag_utils import load_file, process_input_data, postprocess_output, test_kilt
from ck import CK
from vllm import LLM, SamplingParams

def call_ck(model, base_prompts, context_prompts, stop, params_dict):
    
    sequences = model.generate(base_prompts, context_prompts, **params_dict)

    for stop_word in stop:
        length_to_remove = len(stop_word)
        if sequences[-length_to_remove:] == stop_word:
            sequences = sequences[:-length_to_remove]
    output_str = sequences.strip()
    return output_str


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='Models/Llama-3-8B-Instruct')
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--mode', type=str, default='ck', 
                    help='ck, base_rag, base_no_rag')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--adaptive', type=bool, default=True)
    parser.add_argument('--input_file', type=str, 
                        default='work7/datasets/kilt/fever_bge_top100.jsonl')
    parser.add_argument('--retrieval_augment', default=1, action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--max_new_tokens', type=int,
                        default=64)
    parser.add_argument('--max_length', type=int, 
                        default=4096)
    parser.add_argument('--metric', type=str, 
                        default='accuracy')
    parser.add_argument('--top_n', type=int, 
                        default=10,help="number of passages to be considered.")
    parser.add_argument('--task', type=str, 
                        default='fever',help="which task will be used.")
    parser.add_argument('--user_chat_template', action='store_true')
    parser.add_argument('--llama_style', action='store_true',
                        help="whether to use llama.")
    parser.add_argument('--rerank', action='store_true',
                        help="whether to use refinement.")

    parser.add_argument('--output_path', type=str,default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--case_num', type=int, default=20)
    args = parser.parse_args()


    if args.output_path!=None:
        output_path = os.path.join(args.output_path, args.exp_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path=None

    
    params_dict = {
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 100,
            "max_new_tokens": args.max_new_tokens,
            "logprobs": None,
            "mode": args.mode,
            "alpha": 1.0,
            "select_top": 10,
            "adaptive": args.adaptive
        }

    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    
    stop = []
    model = CK(model_name, device, num_gpus, max_gpu_memory=args.max_gpu_memory)
    model.set_stop_words(stop)

    # for top_n in args.top_n:
    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]
    input_data = process_input_data(input_data, args, args.top_n, model.tokenizer)


    final_results = []
    for idx, d in enumerate(tqdm(input_data)):
        pred = call_ck(model, d['instruction'], d['context_instruction'], stop, params_dict)
        d["output"] = pred
        final_results.append(d)

    if output_path is not None:
        output_path = os.path.join(output_path, str(args.task)+'output.jsonl')
        with open(output_path, "w") as f:
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
    print("results are saved in:", output_path)

    for item in input_data:
        metric_result = test_kilt(args.task, 'em', item["output"], item)
        item["em"] = metric_result

    for item in input_data:
        metric_result = test_kilt(args.task, 'rouge', item["output"], item)
        item["rouge"] = metric_result

    for item in input_data:
        metric_result = test_kilt(args.task, 'f1', item["output"], item)
        item["f1"] = metric_result

    for item in input_data:
        metric_result = test_kilt(args.task, 'accuracy', item["output"], item)
        item["accuracy"] = metric_result


    print(args.task)
    print("overall result em: {0}".format(
        np.mean([item["em"] for item in input_data])))
    print("overall result accuracy: {0}".format(
        np.mean([item["accuracy"] for item in input_data])))
    print("overall result rouge: {0}".format(
        np.mean([item["rouge"] for item in input_data])))
    print("overall result f1: {0}".format(
        np.mean([item["f1"] for item in input_data])))
    print("The parameter configuration is as follows:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print('finish')

if __name__ == "__main__":
    main()