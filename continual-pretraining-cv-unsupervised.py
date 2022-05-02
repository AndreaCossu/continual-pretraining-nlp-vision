import os
import avalanche as avl
import torch
from torchvision import datasets
import argparse
from utils import ClassificationPresetEval, ClassificationPresetTrain, split_inaturalist, UnsupBEiTSeqCLF, UnsupBEiTLM, \
    UnsupBEiTNaive, json_save_dict, freeze_model_but_classifier, InMemoryCORe50

parser = argparse.ArgumentParser()
parser.add_argument('--result_folder', type=str, help='folder in which to save models, appended to cache folder')

parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size for continual task')
parser.add_argument('--eval_batch_size', type=int, default=128, help='evaluation batch size for continual task')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for continual task')
parser.add_argument('--epochs', type=int, default=10, help='Training epochs for continual task')
parser.add_argument('--joint_epochs', type=int, default=5, help='Training epochs for downstream task')
parser.add_argument('--joint_lr', type=float, default=1e-3, help='Learning rate for downstream task')
parser.add_argument('--joint_batch_size', type=int, default=256, help='Train and eval batch size for joint training')
parser.add_argument('--joint_decay', type=float, default=0, help='Weight decay for downstream task')
parser.add_argument('--val_perc', type=float, default=0, help='Greater than 0 to build validation set, 0 to use test set'
                                                              'for CORe50')

parser.add_argument('--large', action="store_true", help="use large version of beit")

parser.add_argument('--no_cl', action="store_true", help="only evaluate pretrained model on downstream task")
parser.add_argument('--linear_eval', action="store_true", help="linear evaluation")
parser.add_argument('--load_pretrained', type=str, default="", help="format string to the path of model already pretrained "
                                                                    "format will be filled with experience id")

args = parser.parse_args()


def mycollate(batch):
    x = [el[0] for el in batch]
    labels = torch.tensor([el[1] for el in batch]).long()
    tids = torch.tensor([el[2] for el in batch]).long()
    return [x, labels, tids]


def mycollate_mbatches(mbatches):
    batch = []
    for i in range(len(mbatches[0])):
        if i == 0:
            t = []
            for el in mbatches:
                t += el[0]
        else:
            t = torch.cat([el[i] for el in mbatches], dim=0)
        batch.append(t)
    return batch

input_size = 112  # resize image to this size
core_transforms = {'train': ClassificationPresetTrain(crop_size=input_size, vit=True),
                   'val': ClassificationPresetEval(crop_size=input_size, resize_size=256, vit=True)}


cache_dir = '/ddnbig/a.anonymuser/vision'
data_dir = '/ddnbig/a.anonymuser/data/'

os.makedirs(os.path.join(cache_dir, args.result_folder, 'joint'), exist_ok=True)
n_exps = 5
if not args.no_cl:
    for exp_id in range(n_exps):
        os.makedirs(os.path.join(cache_dir, args.result_folder, 'joint', f'exp_{exp_id}'), exist_ok=True)

json_save_dict(args.__dict__, os.path.join(cache_dir, args.result_folder, 'args.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"

n_classes = 14
if not args.no_cl:
    d = datasets.INaturalist(os.path.join(data_dir, 'inaturalist2018'), download=False, version='2018', target_type='super')
    dtrain, dtest = split_inaturalist(d, None, in_memory=False)
    benchmark = avl.benchmarks.nc_benchmark(dtrain, dtest, n_experiences=n_exps, task_labels=False,
                                            shuffle=False, fixed_class_order=list(range(n_classes)),
                                            per_exp_classes={0: 3, 1: 3, 2: 3, 3: 3, 4: 2})

n_classes_downstream = 50
benchmark_downstream = InMemoryCORe50(scenario='nc', run=0, dataset_root=os.path.join(data_dir, 'core50'),
                                      train_transform=core_transforms['train'], eval_transform=core_transforms['val'],
                                      val_perc=args.val_perc)

f = open(os.path.join(cache_dir, args.result_folder, 'log.txt'), 'w')
fjoint = open(os.path.join(cache_dir, args.result_folder, 'joint', 'log.txt'), 'w')
if (not args.no_cl) and (not args.load_pretrained):
    eval_plugin = avl.training.plugins.EvaluationPlugin(
        avl.evaluation.metrics.loss_metrics(experience=True, stream=True, epoch=True),
        avl.evaluation.metrics.timing_metrics(epoch=True),
        loggers=[avl.logging.InteractiveLogger(),
                 avl.logging.TextLogger(f),
                 avl.logging.TensorboardLogger(os.path.join(cache_dir, args.result_folder))],
        benchmark=benchmark)

eval_plugin_downstream = avl.training.plugins.EvaluationPlugin(
    avl.evaluation.metrics.accuracy_metrics(epoch=True, stream=True),
    avl.evaluation.metrics.loss_metrics(epoch=True, stream=True),
    avl.evaluation.metrics.timing_metrics(epoch=True),
    loggers=[avl.logging.InteractiveLogger(),
             avl.logging.TextLogger(fjoint),
             avl.logging.TensorboardLogger(os.path.join(cache_dir, args.result_folder, 'joint'))],
    benchmark=benchmark_downstream)

if args.large:
    pretrained_path = 'microsoft/beit-large-patch16-224-pt22k'
    pretrained_path_extractor = 'microsoft/beit-large-patch16-224-pt22k'
else:
    pretrained_path = 'microsoft/beit-base-patch16-224-pt22k'
    pretrained_path_extractor = 'microsoft/beit-base-patch16-224-pt22k'
modelname = 'beit'
if args.no_cl:
    model = UnsupBEiTSeqCLF(pretrained_path=pretrained_path, pretrained_path_extractor=pretrained_path_extractor,
                            num_classes=n_classes_downstream, device=device)
else:
    model = UnsupBEiTLM(pretrained_path=pretrained_path, pretrained_path_extractor=pretrained_path_extractor,
                        device=device)
head_name = 'classifier'
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

plugins = []
if (not args.no_cl) and (not args.load_pretrained):
    strategy = UnsupBEiTNaive(model, optimizer, None, train_mb_size=args.train_batch_size,
                              eval_mb_size=args.eval_batch_size, train_epochs=args.epochs, device=device, eval_every=-1,
                              evaluator=eval_plugin, plugins=plugins)

joint_train_mb_size = joint_eval_mb_size = args.joint_batch_size
if args.no_cl:
    freeze_model_but_classifier(model, args.linear_eval, head_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.joint_lr)
    strategy_downstream = avl.training.JointTraining(model, optimizer, torch.nn.CrossEntropyLoss(), train_mb_size=joint_train_mb_size,
                                                     eval_mb_size=joint_eval_mb_size, train_epochs=args.joint_epochs, device=device,
                                                     evaluator=eval_plugin_downstream, plugins=None, eval_every=1)
    strategy_downstream.train(benchmark_downstream.train_stream, eval_streams=[benchmark_downstream.test_stream],
                              num_workers=2)
    model.save_pretrained(os.path.join(cache_dir, args.result_folder, 'joint', f'{modelname}_joint'))
else:
    for exp_id, exp in enumerate(benchmark.train_stream):
        if args.load_pretrained == "":
            strategy.train(exp, num_workers=2, collate_fn=mycollate, collate_mbatches=mycollate_mbatches)
            model.save_pretrained(os.path.join(cache_dir, args.result_folder, f'{modelname}_{exp_id}'))
            strategy.eval(benchmark.test_stream, collate_fn=mycollate)
            model_down = UnsupBEiTSeqCLF(num_classes=n_classes_downstream, pretrained_path_extractor=pretrained_path_extractor,
                                         pretrained_path=os.path.join(cache_dir, args.result_folder, f'{modelname}_{exp_id}'),
                                         device=device)
        else:
            model_down = UnsupBEiTSeqCLF(num_classes=n_classes_downstream, pretrained_path_extractor=pretrained_path_extractor,
                                         pretrained_path=args.load_pretrained.format(exp_id),
                                         device=device)

        freeze_model_but_classifier(model_down, args.linear_eval, head_name)

        optimizer_down = torch.optim.Adam(model_down.parameters(), lr=args.joint_lr, weight_decay=args.joint_decay)

        eval_plugin_downstream = avl.training.plugins.EvaluationPlugin(
            avl.evaluation.metrics.accuracy_metrics(epoch=True, stream=True),
            avl.evaluation.metrics.loss_metrics(epoch=True, stream=True),
            avl.evaluation.metrics.timing_metrics(epoch=True),
            loggers=[avl.logging.InteractiveLogger(),
                     avl.logging.TextLogger(fjoint),
                     avl.logging.TensorboardLogger(os.path.join(cache_dir, args.result_folder,
                                                                'joint', f'exp_{exp_id}'))],
            benchmark=benchmark_downstream)
        strategy_downstream = avl.training.JointTraining(model_down, optimizer_down, torch.nn.CrossEntropyLoss(), train_mb_size=joint_train_mb_size,
                                                         eval_mb_size=joint_eval_mb_size, train_epochs=args.joint_epochs, device=device,
                                                         evaluator=eval_plugin_downstream, plugins=None, eval_every=1)
        strategy_downstream.train(benchmark_downstream.train_stream, eval_streams=[benchmark_downstream.test_stream],
                                  num_workers=2)
        model_down.save_pretrained(os.path.join(cache_dir, args.result_folder, f'{modelname}_downstream_{exp_id}'))


if (not args.no_cl) and (not args.load_pretrained):
    metrics_dict = dict(eval_plugin.get_all_metrics())
    json_save_dict(metrics_dict, os.path.join(cache_dir, args.result_folder, 'metrics.json'))
    f.close()

metrics_dict_joint = dict(eval_plugin_downstream.get_all_metrics())
json_save_dict(metrics_dict_joint, os.path.join(cache_dir, args.result_folder, 'joint', 'metrics.json'))
fjoint.close()
