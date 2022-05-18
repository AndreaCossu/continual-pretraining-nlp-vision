import os
from datasets import Dataset
from utils import filter_by_classes, filtered_classes, create_small_dataset, create_tokenizer

### SPECIFY YOUR PARAMETERS HERE ###
base_data = '/disk3/your_username'
base_save = '/disk2/your_username'
root_path = os.path.join(base_data, 'arxiv_archive-master/processed_data/20200101/per_year')
save_path = os.path.join(base_data, 'arxiv_archive-master/saved')
save_path_generic = os.path.join(base_data, 'emotion')
save_path_small = os.path.join(base_data, 'arxiv_small')
cache_dir = os.path.join(base_save, 'huggingface')
wb_save = os.path.join(base_save, 'wandb')
log_dir = os.path.join(base_save, 'logs')

use_new_tokens = False
use_bert = True
#################### END PARAMETERS ####################

tokenizername = 'roberta-base' if not use_bert else 'bert-base-cased'
tokenizer = create_tokenizer(tokenizername, add_tokens=use_new_tokens)
append_to_save_dir = ''
if use_bert:
    append_to_save_dir += '_bert'
if use_new_tokens:
    append_to_save_dir += '_new_tokens'


def tokenize(examples):
    return tokenizer(examples['abstract'], padding="max_length", truncation=True)

def tokenize_and_save(d, save_full_path):
    d_tk = d.map(tokenize, batched=True, batch_size=1000)
    d_tk.save_to_disk(save_full_path)

# abstract pretrain
tr_d = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path, 'train', 'tokenized', 'pretrain_task_filtered')),
                         filtered_classes)
ts_d = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path, 'test', 'tokenized', 'pretrain_task_filtered')),
                         filtered_classes)
tr_d = create_small_dataset(tr_d, patterns_per_class=10000)
ts_d = create_small_dataset(ts_d, patterns_per_class=1000)

# abstract finetuning
tr_d_f_orig = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path, 'train', 'tokenized', 'finetuning_task_filtered')),
                                filtered_classes)
ts_d_f = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path, 'test', 'tokenized', 'finetuning_task_filtered')),
                           filtered_classes)
tr_d_f = create_small_dataset(tr_d_f_orig, patterns_per_class=10000)
ts_d_f = create_small_dataset(ts_d_f, patterns_per_class=1000)
_, tv_d_f = create_small_dataset(tr_d_f_orig, patterns_per_class=10000,
                                 patterns_per_class_per_test=1000)


print("Tokenizing")
tokenize_and_save(tr_d, os.path.join(save_path_small, 'train', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir))
tokenize_and_save(ts_d, os.path.join(save_path_small, 'test', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir))
tokenize_and_save(tr_d_f, os.path.join(save_path_small, 'train', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
tokenize_and_save(ts_d_f, os.path.join(save_path_small, 'test', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
tokenize_and_save(tv_d_f, os.path.join(save_path_small, 'valid', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
