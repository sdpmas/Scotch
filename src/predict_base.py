from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from model_new import Model
import torch
import pickle
import time
import scann
import numpy as np
import os
import tracemalloc
import torch.nn.functional as f
import sys
sys.path.append('.')

def convert_examples_to_features(context,nl,lang,tokenizer,block_size=510):
    
    nl_tokens=tokenizer.tokenize(nl)[:block_size-2]
    nl_len=len(nl_tokens)
    
    nl_tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)[:block_size]
    # assert len(nl_tokens)==len(nl_ids)
    padding_length = block_size - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    return nl_ids

def main(query,context,searcher,model,codebase,lang,tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    query_ids=torch.tensor(convert_examples_to_features(context,query,lang,tokenizer)).unsqueeze(0).to(device)
    query_repr=f.normalize(model.get_representation_one(query_ids,device=device,mode='query').unsqueeze(0),dim=1).squeeze(0)
    start=time.time()	
    codes,scores=searcher.search(query_repr.detach().cpu().numpy(),pre_reorder_num_neighbors=None,final_num_neighbors=20)
   
    end=time.time()
    
    for i,c in enumerate(codes[:1]):
        print(f'code sample {i} :', codebase[c]['code'])
        print('url: ',codebase[c]['url'],'\n')
        time.sleep(0.1)
        print('-----------------')
  
    print("time elapsed: ",end-start)
    return
    
if __name__=='__main__':
    # query=input('Enter your NL query:')
    lang='python'
    print('current dir: ',os.getcwd())
    context=''
    query="""Language: python Query: Returns the difference of two or more SortedSets as a new SortedSet.

        (i.e. all elements that are in this SortedSet but not the others.)
    """
    
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
    
    codebase= pickle.load(open(f'searchers/scann_searcher_base/codebase.bin','rb'))
   
    searcher = scann.scann_ops_pybind.load_searcher(f'searchers/scann_searcher_base')
  
    main(query=query,context=context,searcher=searcher,model=model,codebase=codebase,lang=lang,tokenizer=tokenizer)	 
    while(True):
        query=input('Enter your NL query:')	
        main(query=query,context=context,searcher=searcher,model=model,codebase=codebase,lang=lang,tokenizer=tokenizer)	
        

