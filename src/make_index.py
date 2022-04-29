import time
import numpy as np
from torch.utils import data
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import sys
sys.path.append('.')
from model import Model
import torch
import pickle
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn.functional as f
from tqdm import tqdm 
import subprocess
import scann
from itertools import repeat
from pathlib import Path
import os
import traceback
import logging
logging.basicConfig(filename='logs/index.log',filemode='w',level=logging.INFO)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                    code_tokens,
                    code_ids,
                    original_code,
                    docstring,url,language,license

    ):
        self.code_tokens = code_tokens
        self.original_code=original_code
        self.code_ids = code_ids
        self.docstring=docstring
        self.url=url
        self.language=language
        self.license=license

def convert_examples_to_features(js,tokenizer,block_size):
    #code
    docstring=js['docstring']
    if not js['identifier']:
        js['identifier'] = 'None'
    code='Function Identifier: '+js['identifier']+'\nFunction Definition:\n'+js['function']  
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = block_size - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length

    original_code=js['function']
    return InputFeatures(code_tokens,code_ids,original_code=original_code,docstring=docstring,url=js['url'],language=js['language'],license=js['license'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size=510, datapath=None,language=None):
        self.examples = []
        data=self.get_data(datapath,language)
        np.random.seed(69)
        np.random.shuffle(data)
        
        for i,js in tqdm(enumerate(data),total=len(data)):
            try:
                converted=convert_examples_to_features(js,tokenizer,block_size)
            except:
                print(traceback.format_exc())
            if converted:
                self.examples.append(converted)

        print('total len: ',len(self.examples))

    def get_data(self,datapath,language):
        file_paths=sorted(Path(os.path.join(datapath,language)).glob('*.bin'))
        data=[]
        for file_path in file_paths:
            loaded_f=pickle.load(open(file_path,'rb'))
            for i,line in enumerate(loaded_f):
                data.append(line)
        return data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),self.examples[i].original_code,self.examples[i].docstring,self.examples[i].url,self.examples[i].language,self.examples[i].license)

def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        # print(result)
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')   

def main(model,tokenizer,dataset_path,language):
    # show_gpu('GPU memory usage initially:')
    language_map={'py':'python','js':'javascript','go':'go','java':'java'}
    logging.info(f'{language}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    dataset=TextDataset(tokenizer,datapath=dataset_path,language=language)
    logging.info(f'{len(dataset)} examples')
    sampler = SequentialSampler(dataset) 
    
    dataloader = DataLoader(dataset, sampler=sampler, 
                                    batch_size=300)

    codebase={}
    idx_count=0
    reprs=[]
    for step, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        codes=batch[0]
        orig=batch[1]
        docstrings=batch[2]
        urls=batch[3]
        languages=batch[4]
        licenses=batch[5]
        
        with torch.no_grad():
            embeds=model.get_representation_batch(qc_ids=codes,device=device,mode='code')
            embeds=embeds.cpu()
            
        for embed,orig_code,docstring,url,lang,license in zip(embeds,orig,docstrings,urls,languages,licenses):
            assert lang==language_map[language]
            codebase[idx_count]={'code':orig_code,'url':url,'language':lang,'license':license}
            reprs.append(embed)
            idx_count+=1
            assert len(reprs)==idx_count
    scann_dataset=torch.stack(reprs)
    normalized_dataset=f.normalize(scann_dataset,dim=1)
    searcher = scann.scann_ops_pybind.builder(normalized_dataset, 15, "dot_product").tree(
        num_leaves=2500, num_leaves_to_search=1000, training_sample_size=500000).score_ah(
        2, anisotropic_quantization_threshold=0.2).build()
    searcher.serialize(f'searchers/final/{language}/')
    logging.info(f'{len(codebase)} examples')
    logging.info('searcher saved')
    pickle.dump(codebase,open(f'searchers/final/{language}/codebase.bin','wb'))
    logging.info('codebase saved')

if __name__=='__main__':
    # specify your language and mode(base or context) here
    language='py'
    dataset_path=f'context_data/dedup'
    model_name='saved_search_context/checkpoint-best-mrr/model.bin'
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
    main(model,tokenizer,dataset_path=dataset_path,language=language)