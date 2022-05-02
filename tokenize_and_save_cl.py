import os
from datasets import Dataset, load_dataset
from utils import filter_by_classes, filtered_classes, create_tokenizer_cl

### SPECIFY YOUR PARAMETERS HERE ###
base_data = '/disk3/anonymuser'
root_path = os.path.join(base_data, 'arxiv_archive-master/processed_data/20200101/per_year')
save_path = os.path.join(base_data, 'arxiv_archive-master/saved')
save_path_generic = os.path.join(base_data, 'emotion')
save_path_generic_qnli = os.path.join(base_data, 'glue', 'qnli', '1.0.0')
save_path_small = os.path.join(base_data, 'arxiv_small')

use_new_tokens = False
use_bert = True
#################### END PARAMETERS ####################


tokenizer_name = 'roberta-base' if not use_bert else 'bert-base-cased'

cl_filtered_classes = [[filtered_classes[i], filtered_classes[i+1]] for i in range(0, len(filtered_classes), 2)]

# generic finetuning emotion
tr_d_g = Dataset.load_from_disk(os.path.join(save_path_generic, 'train_tokenized'))
ts_d_g = Dataset.load_from_disk(os.path.join(save_path_generic, 'test_tokenized'))
tv_d_g = Dataset.load_from_disk(os.path.join(save_path_generic, 'valid_tokenized'))

# generic finetuning qnli
tr_d_qnli = load_dataset("glue", "qnli", split='train', cache_dir=base_data)
tv_d_qnli = load_dataset("glue", "qnli", split='validation', cache_dir=base_data)

for exp_id, exp in enumerate(cl_filtered_classes):
    append_to_save_dir = f'{exp_id}'
    if use_new_tokens:
        append_to_save_dir += '_new_tokens'
    if use_bert:
        append_to_save_dir += '_bert'

    append_to_save_dir_generic = ''
    if use_new_tokens:
        append_to_save_dir_generic += f'{exp_id}_new_tokens'
    if use_bert:
        append_to_save_dir_generic += '_bert'

    variant = '_bert' if use_bert else ''

    tokenizer = create_tokenizer_cl(tokenizer_name, exp_id=exp_id, add_tokens=use_new_tokens)


    def tokenize_abstract(examples):
        return tokenizer(examples['abstract'], padding="max_length", truncation=True)


    def tokenize_generic(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    def tokenize_qnli(examples):
        return tokenizer(examples['question'], examples['sentence'], padding="max_length", truncation=True)


    def tokenize_and_save(d, save_full_path, tkz_f):
        d_tk = d.map(tkz_f, batched=True, batch_size=1000)
        d_tk.save_to_disk(save_full_path)


    # abstract pretrain
    tr_d = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path_small, 'train', 'tokenized', 'pretrain_task_filtered'+variant)),
                             exp)
    ts_d = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path_small, 'test', 'tokenized', 'pretrain_task_filtered'+variant)),
                             exp)

    # abstract finetuning
    tr_d_f = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path_small, 'train', 'tokenized', 'finetuning_task_filtered'+variant)),
                               exp)
    ts_d_f = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path_small, 'test', 'tokenized', 'finetuning_task_filtered'+variant)),
                               exp)
    tv_d_f = filter_by_classes(Dataset.load_from_disk(os.path.join(save_path_small, 'valid', 'tokenized', 'finetuning_task_filtered'+variant)),
                               exp)


    print("Tokenizing")
    tokenize_and_save(tr_d, os.path.join(save_path_small, 'train', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir), tokenize_abstract)
    tokenize_and_save(ts_d, os.path.join(save_path_small, 'test', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir), tokenize_abstract)
    tokenize_and_save(tr_d_f, os.path.join(save_path_small, 'train', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir), tokenize_abstract)
    tokenize_and_save(ts_d_f, os.path.join(save_path_small, 'test', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir), tokenize_abstract)
    tokenize_and_save(tv_d_f, os.path.join(save_path_small, 'valid', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir), tokenize_abstract)

    if use_new_tokens or exp_id == 0:
        tokenize_and_save(tr_d_g, os.path.join(save_path_generic, 'train_tokenized'+append_to_save_dir_generic), tokenize_generic)
        tokenize_and_save(tv_d_g, os.path.join(save_path_generic, 'valid_tokenized'+append_to_save_dir_generic), tokenize_generic)

        tokenize_and_save(tr_d_qnli, os.path.join(save_path_generic_qnli, 'train_tokenized'+append_to_save_dir_generic), tokenize_qnli)
        tokenize_and_save(tv_d_qnli, os.path.join(save_path_generic_qnli, 'valid_tokenized'+append_to_save_dir_generic), tokenize_qnli)
