import os
import gc
import sys
import copy
import psutil
import torch
import gzip
import json
import glob
import random

from collections import Counter
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
from tqdm import tqdm, trange
from transformers import *
from torch.utils.data import Dataset, Sampler

def get_position(entity, text, tokenizer, max_length):
    '''

    Inputs: input text and the answer span
    Outputs: 
        1. start and end position index of the answer
        2. encoded text token ids
        3. Boolean value indicating if the text contains the answer

    '''

    def index_sublist(text, l, entity):
        if entity not in text:
            return (0, 0), False

        sl = tokenizer.encode(entity)[1:-1]
        sl_length = len(sl)
        for i in range(len(l)):
            if l[i] == sl[0] and l[i:i + sl_length] == sl:
                return (i, i + sl_length - 1), True
        return (0, 0), False

    text_tok = tokenizer.encode(text, max_length=max_length)
    position, has_ans = index_sublist(text, text_tok, entity)
    return {
        'position': position,
        'text_tok': text_tok,
        'has_ans': has_ans,
    }

class ContextSampler(object):
    '''

    for each query, randomly sample 1 positive and n negative samples
    for retrieval-extraction pretraining

    '''

    def __init__(self, entity, contexts, tokenizer, max_length):
        self.entity = entity
        self.contexts = contexts
        self.pos_ctx = []
        self.neg_ctx = []
        self.num_pos_ctx = 0
        self.num_neg_ctx = 0

        for context in contexts:

            index = get_position(entity, context, tokenizer, max_length)

            if index['has_ans']:
                self.pos_ctx.append(index)
                self.num_pos_ctx += 1

            if not index['has_ans']:
                self.neg_ctx.append(index)
                self.num_neg_ctx += 1

    def sample(self, num_samples):
        neg_num = num_samples - 1
        label = random.randint(0, neg_num)
        ctx_list = random.sample(self.neg_ctx, neg_num)
        correct_sample = random.sample(self.pos_ctx, 1)
        ctx_list = ctx_list[:label] + correct_sample + ctx_list[label:]
        return ctx_list, label

class RexExample(object):
    '''
    A Rex example for pretraining, containing a query and a context sampler
    Samples data for retrieval and pretraining
    '''

    def __init__(self, q_id, question, answer, contexts, tokenizer,
                ques_max_length=64, ctx_max_length=320, all_max_length=384):
        self.q_id = q_id
        self.question_text = question
        self.answer_text = answer
        self.ques_max_length = ques_max_length
        self.ctx_max_length = ctx_max_length
        self.all_max_length = all_max_length
        self.pad_token_id = tokenizer.pad_token_id

        self.context_sampler = ContextSampler(answer, contexts, tokenizer, ctx_max_length)
        self.ques_ids = tokenizer.encode(question, max_length=ques_max_length)
        self.ques_length = len(self.ques_ids)

        self.num_ques_pads = ques_max_length - self.ques_length
        self.ques_attention_mask = [1] * self.ques_length + [tokenizer.pad_token_id] * self.num_ques_pads

        self.token_type_id = [0] * self.ques_length + [1] * (all_max_length - self.ques_length)
        self.cur_ret_examples = []

    def reset(self):
        self.cur_ret_examples = []

    def gen_attention_mask(self, input_ids, max_length=None, paded=False):
        if paded:
            attention_mask = [int(x != self.pad_token_id) for x in input_ids]
        else:
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
        return attention_mask

    def pad(self, ids, max_length):
        return ids + [self.pad_token_id] * (max_length - len(ids))

    def get_ret_example(self, num_ctx):
        '''

        Sample training targets for retrieval

        '''
        ctx_list, label = self.context_sampler.sample(num_ctx)
        self.cur_ret_examples = ctx_list
        paded_ctx_ids = [self.pad(x['text_tok'], self.ctx_max_length) for x in ctx_list]
        ctx_attention_mask = [self.gen_attention_mask(x, paded=True) for x in paded_ctx_ids]
        return {
            'ques_ids': self.pad(self.ques_ids, self.ques_max_length),
            'ques_attention_mask': self.ques_attention_mask,
            'ctx_ids': paded_ctx_ids,
            'ctx_attention_mask': ctx_attention_mask,
            'label': label
        }

    def get_ext_example(self, ctx_id=None, ctx_obj=None):
        '''

        Sample training targets for extraction based on retrieval results.
        
        '''
        if ctx_id:
            ctx = self.cur_ret_examples[ctx_id]
        elif ctx_obj:
            ctx = ctx_obj
        token_ids = self.pad(self.ques_ids + ctx['text_tok'][1:], self.all_max_length)
        attention_mask = self.gen_attention_mask(token_ids, paded=True)
        start_position = ctx['position'][0]
        end_position = ctx['position'][1]
        if start_position:
            start_position += self.ques_length - 1
            end_position += self.ques_length - 1
        self.reset()
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'start_position': start_position,
            'end_position': end_position,
            'token_type_id': self.token_type_id
        }

    def get_rex_example(self, num_ctx):
        '''

        Sampling training data for combined retrieval-extraction training.

        '''
        ctx_list, label = self.context_sampler.sample(num_ctx)
        examples = [self.get_ext_example(ctx_obj=x) for x in ctx_list]
        return {
            'token_ids': torch.Tensor([x['token_ids'] for x in examples]).long(),
            'attention_mask': torch.Tensor([x['attention_mask'] for x in examples]).long(),
            'start_position': torch.Tensor([x['start_position'] for x in examples]).long(),
            'end_position': torch.Tensor([x['end_position'] for x in examples]).long(),
            'token_type_id': torch.Tensor([x['token_type_id'] for x in examples]).long(),
            'label': label
        }

class RexBatch(object):
    '''

    Combining RexExamples as a batch

    '''

    def __init__(self, rex_examples: list, num_ctx: int, device):
        self.rex_examples = rex_examples
        self.batch_size = len(rex_examples)
        self.device = device
        self.num_ctx = num_ctx
        examples = [rex.get_rex_example(num_ctx) for rex in rex_examples]
        self.data = {
            'input_ids': torch.cat([x['token_ids'] for x in examples], dim=0).to(device),
            'attention_mask': torch.cat([x['attention_mask'] for x in examples], dim=0).to(device),
            'start_positions': torch.cat([x['start_position'] for x in examples], dim=0).to(device),
            'end_positions': torch.cat([x['end_position'] for x in examples], dim=0).to(device),
            'token_type_ids': torch.cat([x['token_type_id'] for x in examples], dim=0).to(device),
            'label': torch.Tensor([x['label'] for x in examples]).long().to(device),
            'num_ctx': num_ctx,
            'batch_size': self.batch_size,
            'idx_base' : torch.range(0, self.batch_size - 1).long().to(device) * num_ctx
        }

    def concat_ques_ctx(self, ctx_pred):
        device = self.device
        ext_examples = [x.get_ext_example(y) for x, y in zip(self.rex_examples, ctx_pred.tolist())]
        output = {
            'input_ids': torch.Tensor([x['token_ids'] for x in ext_examples]).long().to(device),
            'attention_mask': torch.Tensor([x['attention_mask'] for x in ext_examples]).to(device),
            'start_positions': torch.Tensor([x['start_position'] for x in ext_examples]).long().to(device),
            'end_positions': torch.Tensor([x['end_position'] for x in ext_examples]).long().to(device),
            'token_type_ids': torch.Tensor([x['token_type_id'] for x in ext_examples]).long().to(device),
        }
        return output

class RexDataset(Dataset):
    '''

    The dataset for rex examples. Reads raw input files when all data
    in the previous file are used.

    Supports indexing and sequential sampler, does not support random sampler.

    '''

    def __init__(self, data_dir, threads, tokenizer):
        self.tokenizer = tokenizer
        self.threads = threads
        self.fn_list = glob.glob('{}/ssptGen/*/*.gz'.format(data_dir), recursive=True)
        print(len(self.fn_list))
        self.fn_idx = 0
        self.cur_rex_limit = 0
        self.cur_rex_list = []
        self.update_rex_list()
        print('Finished init')

    def epoch_init(self, load=False):
        random.shuffle(self.fn_list)
        self.fn_idx = 0
        if load:
            self.update_rex_list()
        self.rel_idx_base = 0
        self.cur_rex_limit = 0
        self.cur_rex_limit = len(self.cur_rex_list)

    def skip_file(self):
        self.fn_idx += 1
        self.rel_idx_base = self.cur_rex_limit
        self.update_rex_list()

    def update_rex_list(self):
        del self.cur_rex_list
        gc.collect()
        self.cur_rex_list = proc_file(self.fn_list[self.fn_idx], self.threads, self.tokenizer)
        self.cur_rex_limit += len(self.cur_rex_list)

    def __len__(self):
        return len(self.cur_rex_list) * (len(self.fn_list) - 1)

    def __getitem__(self, idx):
        if idx > self.cur_rex_limit - 1:
            print(idx)
            self.skip_file()
        rel_idx = idx - self.rel_idx_base
        return self.cur_rex_list[rel_idx]


def proc_line(rex_line: list):
    '''

    Process lines in the input .jsonlines files

    '''
    line = json.loads(rex_line)
    passages = [p for p in line['passage'] if p]
    if len(passages) < 5:
        return None
    q_id = line['qid']
    question = line['question']
    answer_text = line['answers'][0]
    rex_example = RexExample(
        q_id,
        question.replace('[BLANK]', '[MASK]'),
        answer_text,
        passages,
        tokenizer
    )

    if rex_example.context_sampler.num_pos_ctx > 0 and\
            rex_example.context_sampler.num_neg_ctx > 0:
        return rex_example
    else:
        del rex_example
        return None

def proc_line_init(tokenizer_for_wiki):
    global tokenizer
    tokenizer = tokenizer_for_wiki

def proc_file(fn, threads, tokenizer):
    '''

    Process input .jsonlines files
    
    '''
    in_file = gzip.open(fn)
    jsonls = in_file.readlines()

    with Pool(threads, initializer=proc_line_init, initargs=(tokenizer,)) as p:
        new_rex_list = list(tqdm(p.imap(proc_line, jsonls), total=len(jsonls)))

    rex_list = [rex for rex in new_rex_list if rex]
    print('File {} containes {} Rex examples'.format(fn, len(rex_list)))
    del new_rex_list
    del jsonls
    in_file.close()
    gc.collect()
    return rex_list
