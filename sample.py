import os
from qlora import *
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset, Dataset, load_from_disk
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")


hfparser = transformers.HfArgumentParser((
    ModelArguments, DataArguments, TrainingArguments, GenerationArguments
))
model_args, data_args, training_args, generation_args   , extra_args = \
    hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
args = argparse.Namespace(
    **vars(model_args), **vars(data_args), **vars(training_args)
)

print('parsed args')

# path = '/mnt/data/sonia/datasets/synthetic/adult/may10-2.dat'
# path = 'debug.dat'
dataname = args.dataset.split('/')[-2]
modelname = args.output_dir.split('/')[-1]
path = os.path.join(args.output_dir, 'synth.dat')
os.makedirs(path, exist_ok=True)
print('will save to', path)

tokenizer = AutoTokenizer.from_pretrained('/mnt/data/zoo/llama2/llama2-7b-hf/',
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        )
data_module = make_data_module(tokenizer=tokenizer, args=args)
collator = data_module['data_collator']
print('data loaded')

checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
model, tokenizer = get_accelerate_model(args, checkpoint_dir)
model.config.use_cache = False
model.set_templates(collator.get_templates())
    
print('loaded model')

full_dataset = DatasetDict({})
for f in os.listdir(args.dataset):
    if f.endswith('.json'): continue
    full_dataset[f] = load_from_disk(os.path.join(args.dataset, f))
real = full_dataset['train'].to_pandas().drop(['length'], axis=1)

preds = [ [] for _ in range(real.shape[1]) ]
unmatched = []
batch_size = 50
num_samples = 5000 # real.shape[0]
inputs = collator(batch_size*[{'length': 0}])

print('beginning generation')

for batch in tqdm(range(num_samples//batch_size)):
    _, batch_col_toks = model.generate(**inputs) # batch_size x num_cols x max_column_len
    unmatched.append(batch_col_toks)

    for i, col in enumerate(real.columns):
        options_str = real[col].unique()
        options = tokenizer(options_str.tolist(), add_special_tokens=False, padding='max_length', return_tensors='pt', 
                            max_length=args.generation_config.max_column_len, truncation=True)['input_ids']
        preds_col = options_str[cosine_similarity(batch_col_toks[:, i, :], options).argmax(axis=1)]
        preds[i].extend(preds_col)
        
    if batch % 500 == 0:
        unmatched_np = np.concatenate(unmatched, axis=0)
        unmatched_np.tofile(os.path.join(path, 'raw_np'))
        df = pd.DataFrame(preds).T
        hp = Dataset.from_pandas(df)
        hp.save_to_disk(path)
