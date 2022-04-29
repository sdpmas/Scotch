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
    
    if context:
        context_prompt_tokens=tokenizer.tokenize('Code Context: ')[:max(0,block_size-2-nl_len)]
        context=context.replace('\n',' ')+'\n'
        context_tokens=tokenizer.tokenize(context)[-max(0,(block_size-2)-len(context_prompt_tokens)-len(nl_tokens)):]
        nl_tokens=[tokenizer.cls_token]+context_prompt_tokens+context_tokens+nl_tokens+[tokenizer.sep_token]
    else:
        nl_tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)[:block_size]
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
        # print('license: ',codebase[c]['license'],'\n')
        # print('score: ',scores[i],'\n'*2)
        time.sleep(0.1)
        print('-----------------')
  
    print("time elapsed: ",end-start)
    return
    
if __name__=='__main__':
    # query=input('Enter your NL query:')
    lang='python'
    print('current dir: ',os.getcwd())
    query="""Language: python Query: Returns the difference of two or more SortedSets as a new SortedSet.

        (i.e. all elements that are in this SortedSet but not the others.)
    """
    context="""
import bisect
__all__ = ['SortedSet']
class SortedSet (object):
    def __and__ (self, other):
        return self.intersection(other)
    def __cmp__ (self, other):
        raise ValueError ('cannot compare SortedSets using cmp()')
    def __contains__ (self, elem):
        if len(self) == 0:
            return False
        index = bisect.bisect_left(self.elements, elem)
        if index == len(self) or cmp(self.elements[index], elem):
            return False
        else:
            return True
    def __delitem__ (self, index):
        del self.elements[index]
    def __delslice__ (self, lower, upper):
        del self.elements[lower:upper]
    def __eq__ (self, other):
        if not isinstance(other, SortedSet):
            raise TypeError ('can only compare to a SortedSet')
        return self.elements == other.elements
    def __ge__ (self, other):
        if not isinstance(other, SortedSet):
            return False
        return self.issuperset(other)
    def __getitem__ (self, index):
        if isinstance(index, slice):
            indices = index.indices(len(self))
            return SortedSet([self[i] for i in range(*indices)])
        return self.elements[index]
    def __getslice__ (self, lower, upper):
        return SortedSet(self.elements[lower:upper])
    def __gt__ (self, other):
        if not isinstance(other, SortedSet):
            return False
        return self.issuperset(other) and (self != other)
    def __iand__ (self, other):
        self.intersection_update(other)
    def __init__ (self, iterable=None):
        self.elements = []
        if iterable is not None:
            if isinstance(iterable, SortedSet):
                self.elements = list(iterable.elements)
            else:
                for e in iterable:
                    self.add(e)
    def __ior__ (self, other):
        self.update(other)
    def __isub__ (self, other):
        self.difference_update(other)
    def __iter__ (self):
        return iter(self.elements)
    def __ixor__ (self, other):
        self.symmetric_difference_update(other)
    def __le__ (self, other):
        if not isinstance(other, SortedSet):
            return False
        return self.issubset(other)
    def __len__ (self):
        return len(self.elements)
    def __lt__ (self, other):
        if not isinstance(other, SortedSet):
            return False
        return self.issubset(other) and (self != other)
    def __ne__ (self, other):
        if not isinstance(other, SortedSet):
            raise TypeError ('can only compare to a SortedSet')
        return self.elements != other.elements
    def __or__ (self, other):
        return self.union(other)
    def __rand__ (self, other):
        return self & other
    def __repr__ (self):
        return '{self.__class__.__name__}({self.elements!r})'.format(self=self)
    def __reversed__ (self):
        return reversed(self.elements)
    def __ror__ (self, other):
        return self | other
    def __rsub__ (self, other):
        return other.difference(self)
    def __rxor__ (self, other):
        return self ^ other
    def __sub__ (self, other):
        return self.difference(other)
    def __xor__ (self, other):
        return self.symmetric_difference(other)
    def add (self, elem):
        if len(self) == 0:
            self.elements.append(elem)
        index = bisect.bisect_left(self.elements, elem)
        if index == len(self):
            self.elements.append(elem)
        elif cmp(self.elements[index], elem):
            self.elements.insert(index, elem)
        else:
            self.elements[index] = elem
    def clear (self):
        self.elements = []
    def copy (self):
        return SortedSet(self)
  
    """
    # query='print the name of the user'
    
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
    
    codebase= pickle.load(open(f'searchers/scann_searcher_context/codebase.bin','rb'))
   
    searcher = scann.scann_ops_pybind.load_searcher(f'searchers/scann_searcher_context')
  
    main(query=query,context=context,searcher=searcher,model=model,codebase=codebase,lang=lang,tokenizer=tokenizer)
    while(True):
        query=input('Enter your NL query:')	
        main(query=query,context=context,searcher=searcher,model=model,codebase=codebase,lang=lang)	
   