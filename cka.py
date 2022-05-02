import timm
import torch.cuda
import os
import argparse
import torchvision
from tqdm import tqdm
from torch_cka import CKA
from torch.utils.data import DataLoader
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification

from utils import InMemoryCORe50, ClassificationPresetTrain, ClassificationPresetEval, load_qnli, \
    UnsupBEiTSeqCLF, pickle_save_dict, load_generic_finetuning


class MyCKA(CKA):
    """https://github.com/AntixK/PyTorch-Model-Compare/blob/main/torch_cka/cka.py"""
    def __init__(self, modelname, model1, model2, model1_name=None, model2_name=None, model1_layers=None, model2_layers=None,
                 device=torch.device('cpu'), epochs=1):

        super().__init__(model1, model2, model1_name, model2_name, model1_layers, model2_layers, device)
        self.epochs = epochs
        self.modelname = modelname
        self.vision = modelname in ['vit', 'beit', 'resnet']

    @torch.no_grad()
    def compare(self, dataloader) -> None:
        self.model1_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = len(dataloader) * self.epochs

        for epoch in range(self.epochs):
            for minibatch in tqdm(dataloader, desc="| Comparing features |", total=int(num_batches/self.epochs)):
                self.model1_features = {}
                self.model2_features = {}

                if self.vision:
                    x = minibatch[0]
                    if isinstance(x, torch.Tensor):
                        x = x.to(self.device)
                    _ = self.model1(x)
                    _ = self.model2(x)
                else:
                    ids, mask = minibatch['input_ids'], minibatch['attention_mask']
                    tids = minibatch['token_type_ids'] if 'token_type_ids' in minibatch else None
                    ids = ids.to(self.device)
                    mask = mask.to(self.device)
                    if tids is not None:
                        tids = tids.to(self.device)
                    _ = self.model1(input_ids=ids, attention_mask=mask, token_type_ids=tids)
                    _ = self.model2(input_ids=ids, attention_mask=mask, token_type_ids=tids)

                for i, (name1, feat1) in enumerate(self.model1_features.items()):
                    if isinstance(feat1, tuple) and len(feat1) == 1:
                        feat1 = feat1[0].flatten(1)
                    assert isinstance(feat1, torch.Tensor), f"{name1} not a tensor"
                    X = feat1.flatten(1)

                    K = X @ X.t()

                    K.fill_diagonal_(0.0)
                    self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches
                    for j, (name2, feat2) in enumerate(self.model2_features.items()):
                        if isinstance(feat2, tuple) and len(feat2) == 1:
                            feat2 = feat2[0].flatten(1)
                        assert isinstance(feat2, torch.Tensor), f"{name2} not a tensor"
                        Y = feat2.flatten(1)
                        L = Y @ Y.t()
                        L.fill_diagonal_(0)
                        assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                        self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                        self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_results(hsic_matrix, model1name, model2name, save_path: str = None, exp_id: str = None):
    fig, ax = plt.subplots()
    im = ax.imshow(hsic_matrix, origin='lower', cmap='magma')
    ax.set_xlabel(f"Layers {model2name}", fontsize=15)
    ax.set_ylabel(f"Layers {model1name}", fontsize=15)

    add_colorbar(im)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'matrix{exp_id}.png'), dpi=300)
    else:
        plt.show()


def load_model(modelname, path, device=torch.device('cpu'), large=False, qnli=False):
    if modelname == 'roberta' or modelname == 'bert':
        num_labels = 2 if qnli else 6
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    elif modelname == 'resnet':
        model = torchvision.models.resnet101(pretrained=True)
        model.fc = torch.nn.Linear(2048, 50)
        model.load_state_dict(torch.load(path))
    elif modelname == 'vit':
        name = 'vit_base_patch32_224_in21k' if not large else 'vit_large_patch32_224_in21k'
        model = timm.create_model(name, num_classes=50, pretrained=True)
        model.head = torch.nn.Linear(768, 50)
        model.load_state_dict(torch.load(path))
    elif modelname == 'beit':
        pretrained_path_extractor = 'microsoft/beit-base-patch16-224-pt22k' if not large else 'microsoft/beit-large-patch16-224-pt22k'
        model = UnsupBEiTSeqCLF(pretrained_path=path, pretrained_path_extractor=pretrained_path_extractor,
                                num_classes=50, device=device)
    else:
        raise ValueError("Modelname not recognized.")
    model.eval()
    return model


model2layers = {
    'beit': [f'vit.beit.encoder.layer.{l}' for l in range(12)] + ['vit.beit.embeddings.patch_embeddings'],
    'vit': [f'blocks.{l}' for l in range(12)] + ['patch_embed'],
    'resnet': [f'layer1.{l}.bn{j}' for l in range(3) for j in range(1, 4)] +
              [f'layer2.{l}.bn{j}' for l in range(4) for j in range(1, 4)] +
              [f'layer3.{l}.bn{j}' for l in range(23) for j in range(1, 4)] +
              [f'layer4.{l}.bn{j}' for l in range(3) for j in range(1, 4)],
    'bert': [f'bert.encoder.layer.{l}' for l in range(12)] + ['bert.embeddings.word_embeddings'],
    'roberta': [f'roberta.encoder.layer.{l}' for l in range(12)] + ['roberta.embeddings.word_embeddings']
}

model2layers_large = {
    'beit': [f'vit.beit.encoder.layer.{l}' for l in range(24)] + ['vit.beit.embeddings.patch_embeddings'],
    'vit': [f'blocks.{l}' for l in range(24)] + ['patch_embed'],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath_joint', type=str, default='roberta-base', help='model name or path to pretrained model')
    parser.add_argument('--modelpath_cl', type=str, default='roberta-base', help='model name or path to pretrained model')

    parser.add_argument('--modelname', type=str, choices=['roberta', 'bert', 'resnet', 'vit', 'beit'], help='type of model')

    parser.add_argument('--exp_id', type=int, help='experience id of model cl')

    parser.add_argument('--result_folder', type=str, help='folder in which to save results')

    parser.add_argument('--qnli', action="store_true", help='use qnli models for bert and roberta')
    parser.add_argument('--large', action="store_true", help='use large version of beit and vit')

    parser.add_argument('--batch_size', type=int, default=256, help='training batch size'),
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    args = parser.parse_args()

    task_type = 'nlp' if args.modelname in ['roberta', 'bert'] else 'vision'
    model1name = args.modelname+'_joint'
    model2name = args.modelname+str(args.exp_id)
    model1_layers = model2layers[args.modelname] if not args.large else model2layers_large[args.modelname]
    model2_layers = model2layers[args.modelname] if not args.large else model2layers_large[args.modelname]

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cache_dir = f'/ddnbig/a.cossu/{task_type}'
    data_dir = '/ddnbig/a.cossu/data/'
    os.makedirs(os.path.join(cache_dir, args.result_folder), exist_ok=True)
    save_path = os.path.join(cache_dir, args.result_folder)

    model1 = load_model(args.modelname, args.modelpath_joint, device, large=args.large, qnli=args.qnli)
    model2 = load_model(args.modelname, args.modelpath_cl, device, large=args.large, qnli=args.qnli)

    if task_type == 'vision':
        vit = not args.modelname == 'resnet'
        input_size = 112 if args.modelname == 'beit' else 224
        core_transforms = {'train': ClassificationPresetTrain(crop_size=input_size, vit=vit),
                           'val': ClassificationPresetEval(crop_size=input_size, resize_size=256, vit=vit)}

        benchmark = InMemoryCORe50(scenario='nc', run=0, dataset_root=os.path.join(data_dir, 'core50'),
                                   train_transform=core_transforms['train'], eval_transform=core_transforms['val'],
                                   only_test=True)
        dataset = benchmark.test_stream[0].dataset
    else:
        if args.qnli:
            _, dataset = load_qnli(use_bert=(args.modelname == 'bert'))
        else:
            _, dataset = load_generic_finetuning(use_test=True, add_tokens=False,
                                                 exp_id=args.exp_id, use_bert=(args.modelname == 'bert'))
        dataset.set_format(type="torch")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    cka = MyCKA(args.modelname, model1, model2,
                model1_name=model1name,
                model2_name=model2name,
                model1_layers=model1_layers,
                model2_layers=model2_layers,
                epochs=args.epochs,
                device=device)

    cka.compare(dataloader)
    results = cka.export()
    pickle_save_dict(results, os.path.join(save_path, f'cka{args.exp_id}.pickle'))
    plot_results(hsic_matrix=results["CKA"], model1name=model1name, model2name=model2name, exp_id=args.exp_id,
                 save_path=save_path)
