from __future__ import absolute_import, division, print_function

import logging
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

from tqdm import tqdm, trange
import multiprocessing
import pickle 
import random 
from pathlib import Path
import re
from io import StringIO
import  tokenize
import sys
sys.path.append(os.path.dirname(__file__))
from model import Model
from run_base import remove_comments_and_docstrings,set_seed
from run_base import get_all_data,get_data
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logging.basicConfig(level=logging.INFO,filename='logs/eval_base.log',filemode='w')
logger = logging
set_seed(69)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

class InputFeaturesBase(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,language
                
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.language=language
        

def convert_examples_to_features(js,tokenizer,args,remove_comments=None):
    #code
    if not js['identifier']:
        js['identifier'] = 'None'
    
    code_function=js['function'].replace('\n',' ')
    code='Function Identifier: '+js['identifier']+'\nFunction Definition:\n'+code_function
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    
    nl=f"Language: {js['language']}"+' Query: '+js['docstring']
    
    nl_tokens=tokenizer.tokenize(nl)[:args.block_size-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.block_size - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    return InputFeaturesBase(code_tokens,code_ids,nl_tokens,nl_ids,js['url'],js['language'])


class TextDatasetBase(Dataset):
    def __init__(self, tokenizer,data, args):
        # code_data=data[:100000]
        
        self.examples = []
        for i,line in tqdm(enumerate(data),total=len(data)):
            # if i>5000:break
            converted=convert_examples_to_features(line,tokenizer,args)
            if converted:
                self.examples.append(converted)
        np.random.shuffle(self.examples)
        for idx, example in enumerate(self.examples[:20]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("url: {}".format(example.url))
            logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
            logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
            logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
            logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids),self.examples[i].url)
            
def test(args, model, tokenizer,test_languages,all_data=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('language: {}'.format(test_languages[0])) 
    language_map={'python':'py','javascript':'js','go':'go','java':'java'}
    test_data=get_all_data(args.data_path,mode='test')
    test_dataset = TextDatasetBase(tokenizer=tokenizer,data=test_data,args=args)
    
    if not all_data:
        test_dataset.examples=[t for t in test_dataset.examples if language_map[t.language] in test_languages]
        # np.random.shuffle(test_dataset.examples)
    print('final length: ',len(test_dataset))
    if not all_data:
        for e in test_dataset.examples:
            assert language_map[e.language] in test_languages
    logging.info('eval batch size: {}'.format(args.eval_batch_size))
    logging.info('total batch size: {}'.format(args.total_batch_size))
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.total_batch_size)
    model.to(device)
    # multi-gpu evaluate
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    ranks=[]
    for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        code_inputs = batch[0]
        nl_inputs = batch[1]
        urls=batch[2]
        logger.info('------------')
        logger.info(urls[0])
        logger.info(urls[-1])
        logger.info('------------')
        
        with torch.no_grad():
            code_vecs=[]
            nl_vecs=[]
            for i in range(0,len(code_inputs),args.eval_batch_size):
                lm_loss,code_vec,nl_vec = model(code_inputs[i:i+args.eval_batch_size].to(device),nl_inputs[i:i+args.eval_batch_size].to(device))
                # print('inside shape: ',code_vec.shape,nl_vec.shape)
                code_vecs.append(code_vec)
                nl_vecs.append(nl_vec)
            code_vecs=torch.cat(code_vecs,dim=0)
            nl_vecs=torch.cat(nl_vecs,dim=0)
            # print('code_vecs shape: ',code_vecs.shape,' nl_vecs shape: ',nl_vecs.shape)
            eval_loss += lm_loss.mean().item()
            code_vecs=code_vecs.cpu().numpy()
            nl_vecs=nl_vecs.cpu().numpy()
            scores=np.matmul(nl_vecs,code_vecs.T)
            for i in range(len(scores)):
                score=scores[i,i]
                rank=1
                for j in range(len(scores)):
                    if i!=j and scores[i,j]>=score:
                        rank+=1
                ranks.append(1/rank)      
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_mrr=np.mean(ranks)
    perplexity = torch.tensor(eval_loss)
    print('----------eval results------')
    print('language: ',test_languages)
    logger.info('language: {}'.format(test_languages[0]))
    print('eval_loss: ',float(perplexity))
    print('eval_mrr: ',float(np.mean(ranks)))
    logger.info("  Perplexity = %f", perplexity)
    
    logger.info('MRR: '+str(float(eval_mrr)))


def parser_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path", default='context_data/split', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=510, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--output_dir", default='search/out', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_batch_size", default=1000, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--total_batch_size", default=1000, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    return parser.parse_args()

if __name__=='__main__':
    model_name='saved_search_base/checkpoint-best-mrr/model.bin'
    config = RobertaConfig.from_pretrained('codebert',
                                          cache_dir= None)
    tokenizer = RobertaTokenizer.from_pretrained('codebert',
                                                cache_dir=None)
    code_model = RobertaModel.from_pretrained('codebert',
                                            config=config,
                                           cache_dir=None) 
    nl_model = RobertaModel.from_pretrained('codebert',
                                            config=config,
                                           cache_dir=None) 
    model=Model(nl_model,code_model,config,tokenizer,args=None)
    model.load_state_dict(torch.load(model_name))
    
    args=parser_parse()
    all_languages=['all','py','js','go','java']
    for lang in all_languages:
        test(model=model,tokenizer=tokenizer,args=args,test_languages=[lang],all_data=lang=='all')