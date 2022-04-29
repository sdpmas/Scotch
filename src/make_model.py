from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from model import Model
import torch
import scann
import numpy as np
import torch.nn.functional as f
import sys
sys.path.append('.')

if __name__=='__main__':
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
    #search: save a pytorch model pth
    pytorch_total_params = sum(p.numel() for p in nl_model.parameters() if p.requires_grad)
    print('Total number of parameters:', pytorch_total_params)

    torch.save(model.state_dict(), 'saved_search_context/model.pth')
