import os
import avalanche as avl
import torch
import argparse
from transformers import AutoModelForSequenceClassification
from utils import filtered_classes, cache_dir, create_abstracts_avalanche_benchmark, \
    create_tokenizer_cl, HGBaseStrategy, HGJointTraining, HGStreamingLDA, HGFeatureExtractorBackbone

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action="store_true", help='do not use GPU')
parser.add_argument('--tokenizername', type=str, default='', help='if empty, equal to modelname')
parser.add_argument('--modelname', type=str, default='roberta-base', help='huggingface model name or path to pretrained model folder'
                                                                             'to use it for finetuning')

parser.add_argument('--test_on_test', action="store_true", help='eval on test set, otherwise on validation set (only for finetuning)')
parser.add_argument('--add_tokens', action="store_true", help='add domain-specific tokens to tokenizer')

parser.add_argument('--result_folder', type=str, help='folder in which to save models, appended to cache folder')

parser.add_argument('--strategy', type=str, default='naive', choices=['naive', 'cwr', 'replay', 'dslda', 'joint'],
                    help='type of strategy to use')
parser.add_argument('--mem_size', type=int, default=100, help='replay memory size')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size'),
parser.add_argument('--eval_batch_size', type=int, default=8, help='evaluation batch size')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
args = parser.parse_args()

os.makedirs(os.path.join(cache_dir, args.result_folder), exist_ok=True)

num_experiences = 5
cl_filtered_classes = [[filtered_classes[i], filtered_classes[i+1]] for i in range(0, len(filtered_classes), 2)]

modelname = f'{args.modelname}_0' if args.modelname.startswith('/') else args.modelname
model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2*len(filtered_classes[:num_experiences]))
tokenizer = create_tokenizer_cl(args.tokenizername, 0, args.add_tokens)
model.resize_token_embeddings(len(tokenizer))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
device = "cuda" if torch.cuda.is_available() else "cpu"
benchmark = create_abstracts_avalanche_benchmark(cl_filtered_classes, args.add_tokens, args.test_on_test)

f = open(os.path.join(cache_dir, args.result_folder, 'log.txt'), 'w')
eval_plugin = avl.training.plugins.EvaluationPlugin(
    avl.evaluation.metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    avl.evaluation.metrics.loss_metrics(epoch=True, experience=True, stream=True),
    avl.evaluation.metrics.timing_metrics(epoch=True),
    avl.evaluation.metrics.forgetting_metrics(experience=True, stream=True),
    loggers=[avl.logging.InteractiveLogger(),
             avl.logging.TextLogger(f),
             avl.logging.TensorboardLogger(os.path.join(cache_dir, args.result_folder))],
    benchmark=benchmark,
    strict_checks=False
)

plugins = []
if args.strategy == 'cwr' or args.strategy == 'naive' or args.strategy == 'replay':
    if args.strategy == 'cwr':
        plugins.append(avl.training.plugins.CWRStarPlugin(model, freeze_remaining_model=False))
    elif args.strategy == 'replay':
        plugins.append(avl.training.plugins.ReplayPlugin(mem_size=args.mem_size))
    strategy = HGBaseStrategy(model, optimizer, torch.nn.CrossEntropyLoss(), train_mb_size=args.train_batch_size,
                              eval_mb_size=args.eval_batch_size, train_epochs=args.epochs, device=device, eval_every=-1,
                              evaluator=eval_plugin, plugins=plugins)
elif args.strategy == 'joint':
    strategy = HGJointTraining(model, optimizer, torch.nn.CrossEntropyLoss(), train_mb_size=args.train_batch_size,
                               eval_mb_size=args.eval_batch_size, train_epochs=args.epochs, device=device,
                               evaluator=eval_plugin, plugins=plugins)
elif args.strategy == 'dslda':
    model = HGFeatureExtractorBackbone(model.to(device), 'classifier.dense').eval()
    strategy = HGStreamingLDA(model, torch.nn.CrossEntropyLoss(), 768, 2*len(filtered_classes[:num_experiences]),
                              output_layer_name=None, device=device, plugins=plugins, evaluator=eval_plugin, eval_every=-1)
else:
    raise ValueError("Wrong strategy name.")

for exp_id, exp in enumerate(benchmark.train_stream):

    if args.strategy == 'joint':
        strategy.train(benchmark.train_stream)
    else:
        strategy.train(exp)
    strategy.eval(benchmark.test_stream)

    if args.strategy != 'dslda':
        print("Saving finetuned model after experience ", exp_id)
        model.save_pretrained(os.path.join(cache_dir, args.result_folder, f'{os.path.split(args.modelname)[-1]}_finetuned_{exp_id}'))
        print("Finetuned model saved after experience ", exp_id)
    if args.strategy == 'joint':
        break
f.close()
