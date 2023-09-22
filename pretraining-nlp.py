import os
import argparse
import wandb
import random
from torch.nn import Identity
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM
from datasets import Dataset
from utils import filtered_classes, cache_dir, save_path_small, \
    pretrain_model, finetune_model, create_tokenizer, remap_classes, select_informative_examples, \
    freeze_model_but_classifier, CustomRobertaClassificationHead

parser = argparse.ArgumentParser()
parser.add_argument('--log_every', type=int, default=0, help='Step every which log, 0 to log every epoch, -1 to disable')
parser.add_argument('--no_cuda', action="store_true", help='do not use GPU')
parser.add_argument('--eval_every', type=int, default=0, help='Step every which eval, 0 to eval every epoch, -1 to disable')
parser.add_argument('--tokenizername', type=str, default='', help='if empty, equal to modelname')
parser.add_argument('--modelname', type=str, default='roberta-base', help='huggingface model name or path to pretrained model folder'
                                                                             'to use it for finetuning')

parser.add_argument('--pretrain_selection', type=str, default='none', choices=['none', 'random', 'top', 'bottom', 'median'],
                    help='during pretrain, select a subset of the examples according to the loss value.')
parser.add_argument('--num_informative_examples', type=int, default=10000, help='Number of informative examples to select per experience.')

parser.add_argument('--result_folder', type=str, help='folder in which to save models, appended to cache folder')

parser.add_argument('--test_on_test', action="store_true", help='eval on test set, otherwise on validation set (only for finetuning)')
parser.add_argument('--linear_eval', action="store_true", help='use linear evaluation by fixing feature extractor.')

parser.add_argument('--add_tokens', action="store_true", help='add domain-specific tokens to tokenizer')

parser.add_argument('--no_save', action="store_true", help='do not save final model')
parser.add_argument('--only_eval', action="store_true", help='only perform a round of evaluation')

parser.add_argument('--task_type', type=str, default='pretrain', choices=['pretrain', 'finetune'], help='type of task to perform')
parser.add_argument('--train_batch_size', type=int, default=25, help='training batch size')
parser.add_argument('--eval_batch_size', type=int, default=25, help='evaluation batch size')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
args = parser.parse_args()


os.makedirs(os.path.join(cache_dir, args.result_folder), exist_ok=True)

use_bert = True if args.tokenizername == 'bert-base-cased' else False
head_name = ['classifier'] if use_bert else ['classifier.out_proj']

if args.eval_every == -1:
    eval_strategy = 'no'
elif args.eval_every == 0:
    eval_strategy = 'epoch'
else:
    eval_strategy = 'steps'
if args.log_every == -1:
    log_strategy = 'no'
elif args.log_every == 0:
    log_strategy = 'epoch'
else:
    log_strategy = 'steps'

append_to_save_dir = ''
if args.tokenizername == 'bert-base-cased':
    append_to_save_dir += '_bert'
if args.add_tokens:
    append_to_save_dir += '_new_tokens'

project_name = 'huggingface'

tokenizer = create_tokenizer(args.tokenizername, args.add_tokens)
if args.task_type == 'pretrain':
    model = AutoModelForMaskedLM.from_pretrained(args.modelname)
    model.resize_token_embeddings(len(tokenizer))

    tr_d = Dataset.load_from_disk(os.path.join(save_path_small, 'train', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir))
    ts_d = Dataset.load_from_disk(os.path.join(save_path_small, 'test', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir))

    tr_d = tr_d.remove_columns(['primary_cat', 'abstract', 'created'])
    ts_d = ts_d.remove_columns(['primary_cat', 'abstract', 'created'])
    tr_d.set_format(type="torch")
    ts_d.set_format(type="torch")

    if args.pretrain_selection != 'none':
        assert len(tr_d) >= args.num_informative_examples
        device = 'cpu' if args.no_cuda else 'cuda'
        print('Selecting informative examples...')
        tr_d = select_informative_examples(tr_d, model.to(device), device, n_samples=args.num_informative_examples,
                                           mode=args.pretrain_selection)
        print('Done.')
        assert len(tr_d) == args.num_informative_examples

    with wandb.init(project=project_name, name=args.result_folder):
        pretrain_model(args=args, tr_d=tr_d, ts_d=ts_d, model=model, tokenizer=tokenizer, log_strategy=log_strategy,
                       eval_strategy=eval_strategy, eval_only=args.only_eval)
    if (not args.no_save) and (not args.only_eval):
        print("Saving final model")
        model.save_pretrained(os.path.join(cache_dir, args.result_folder, f'{os.path.split(args.modelname)[-1]}_pretrained'))
        print("Model saved")

elif args.task_type == 'finetune':
    model = AutoModelForSequenceClassification.from_pretrained(args.modelname, num_labels=len(filtered_classes))
    model.resize_token_embeddings(len(tokenizer))

    if args.linear_eval:
        if use_bert:
            model.dropout = Identity()
            model.bert.pooler.dense = Identity()
            model.bert.pooler.activation = Identity()
        else:
            model.classifier = CustomRobertaClassificationHead(hidden_size=768, num_labels=10)

    freeze_model_but_classifier(model, args.linear_eval, head_name)

    tr_d = Dataset.load_from_disk(os.path.join(save_path_small, 'train', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
    if args.test_on_test:
        ts_d = Dataset.load_from_disk(os.path.join(save_path_small, 'test', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
    else:
        ts_d = Dataset.load_from_disk(os.path.join(save_path_small, 'valid', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))

    tr_d = tr_d.map(remap_classes)
    ts_d = ts_d.map(remap_classes)

    tr_d = tr_d.remove_columns(['abstract', 'created']).rename_column('primary_cat', 'labels')
    ts_d = ts_d.remove_columns(['abstract', 'created']).rename_column('primary_cat', 'labels')
    tr_d.set_format(type="torch")
    ts_d.set_format(type="torch")

    with wandb.init(project=project_name, name=args.result_folder):
        finetune_model(args=args, tr_d=tr_d, ts_d=ts_d, model=model, log_strategy=log_strategy,
                       eval_strategy=eval_strategy, eval_only=args.only_eval)
    if (not args.no_save) and (not args.only_eval):
        print("Saving final model")
        model.save_pretrained(os.path.join(cache_dir, args.result_folder, f'{os.path.split(args.modelname)[-1]}_finetuned'))
        print("Model saved")
else:
    raise ValueError("Wrong task type!")
