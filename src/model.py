# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

    
class Model(nn.Module):   
    def __init__(self, nl_encoder,code_encoder,config,tokenizer,args=None):
        super(Model, self).__init__()
        self.nl_encoder = nl_encoder
        self.code_encoder=code_encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args

    def get_representation_batch(self,qc_ids=None,device=None,mode='query'):
        """get represenations in batch for either queries or codes"""
        if mode=='query':
            return self.nl_encoder(qc_ids.to(device),attention_mask=qc_ids.ne(1).to(device))[1]
        elif mode=='code':
            return self.code_encoder(qc_ids.to(device),attention_mask=qc_ids.ne(1).to(device))[1]

    def get_representation_one(self,query,device=None,mode='query'):
        """get representation for a single query: less dataset stuffs."""
        with torch.no_grad():
            if mode=="query":
                # print('we are in query mode')
                return self.nl_encoder(query,attention_mask=query.ne(1).to(device))[1].squeeze(dim=0)
            elif mode=="code":
                query_tokens=[self.tokenizer.cls_token]+self.tokenizer.tokenize(query)[:508]+[self.tokenizer.sep_token]
                query_ids=torch.tensor(self.tokenizer.convert_tokens_to_ids(query_tokens)).unsqueeze(dim=0).to(device)
                return self.code_encoder(query_ids,attention_mask=query_ids.ne(1).to(device))[1].squeeze(dim=0)
            elif mode=='code_ids':
                return self.code_encoder(query,attention_mask=query.ne(1).to(device))[1].squeeze(dim=0)

    def forward(self, code_inputs,nl_inputs,return_vec=False): 
        bs=code_inputs.shape[0]
        code_vec=self.code_encoder(code_inputs,attention_mask=code_inputs.ne(1).to(code_inputs.device))[1]
        nl_vec=self.nl_encoder(nl_inputs,attention_mask=nl_inputs.ne(1).to(nl_inputs.device))[1]
        
        if return_vec:
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_vec,nl_vec

      
        
 
