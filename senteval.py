import sys
sys.path.insert(0, 'SentEval')
import senteval
import torch
from transformers import AutoModelForSequenceClassification
from utils import create_tokenizer_cl
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--tokenizername', type=str, default='', help='if empty, equal to modelname')
parser.add_argument('--modelname', type=str, default='roberta-base', help='huggingface model name or path to pretrained model folder'
                                                                             'to use it for finetuning')
parser.add_argument('--add_tokens', action="store_true", help='add domain-specific tokens to tokenizer')
parser.add_argument('--exp_id', type=int, default=0, choices=list(range(5)),
                    help='experience id to load model and tokenizer')
parser.add_argument('--transfer', action="store_true", help='use transfer tasks, else use probing tasks.')
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.modelname)
tokenizer = create_tokenizer_cl(args.tokenizername, args.exp_id, args.add_tokens)
model.resize_token_embeddings(len(tokenizer))


def mean_pooling(model_output, attention_mask):
    """Taken from sentence-transformers python package documentation"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def batcher(params, batch):
    newbatch = [' '.join(sent) for sent in batch]
    encoded = tokenizer(newbatch, truncation=True, padding=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        embeddings = mean_pooling(model(encoded['input_ids'], encoded['attention_mask'], output_hidden_states=True)['hidden_states'][-1],
                                  encoded['attention_mask'])
    return embeddings.numpy()


nhid = 0 if args.transfer else 50
params = {'task_path': 'SentEval/data', 'usepytorch': True, 'kfold': 10, 'cudaEfficient': True}
params['classifier'] = {'nhid': nhid, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                  'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                  'SICKEntailment']

se = senteval.engine.SE(params, batcher)
results = se.eval(transfer_tasks) if args.transfer else se.eval(probing_tasks)
print(results)
