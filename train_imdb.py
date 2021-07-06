import torch
import torch.nn as nn
import torchvision
from datasets import imdb
import numpy as np
import argparse
import os

import utils
import wandb
from tqdm import tqdm
from torchvision import transforms
import torchdata as td
from multiprocessing import Manager
from sklearn.model_selection import StratifiedKFold

class pattern_norm(torch.nn.Module):
    def __init__(self, scale = 1.0):
        super(pattern_norm, self).__init__()
        self.scale = scale

    def forward(self, input):
        sizes = input.size()
        if len(sizes) > 2:
            input = input.view(-1, np.prod(sizes[1:]))
            input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
            input = input.view(sizes)
        return input

class ResNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.feature_pos = 'post'

    def forward(self, x: torch.Tensor, logits_only=False) -> torch.Tensor:
        x = self.m.conv1(x)
        x = self.m.bn1(x)
        x = self.m.relu(x)
        x = self.m.maxpool(x)

        x = self.m.layer1(x)
        x = self.m.layer2(x)
        x = self.m.layer3(x)
        x = self.m.layer4(x)

        feats = x
        x = self.m.avgpool(x)
        x = torch.flatten(x, 1)
        if self.feature_pos == 'post':
            feats = x
        x = self.m.fc(x)

        if logits_only:
            return x
        return x, feats

def resnet18(n_classes):
    model = torchvision.models.resnet18(pretrained='imagenet')
    model.layer4[1].relu = torch.nn.Tanh()
    model.avgpool = torch.nn.Sequential(
        model.avgpool,
        pattern_norm()
    )
    model.fc = torch.nn.Linear(in_features=512, out_features=n_classes, bias=True)
    return ResNetWrapper(model)


def load_imdb(config, shuffle=True):
    cache_p = float(os.environ.get('DATASET_CACHE', 0.50))
    print(f'=> Caching {cache_p*100:.2f}% of data')

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    T_resize = transforms.Resize((224, 224))

    T_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    T_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset_src = imdb.IMDB(root=f'{os.path.expanduser("~")}/data', 
                                    train=True, 
                                    split=config.split, 
                                    target=config.target_attr)
    train_dataset = td.datasets.WrapDataset(train_dataset_src)
    train_dataset.map(td.maps.To(T_resize, 0))
    train_dataset.cache(td.modifiers.UpToPercentage(cache_p, len(train_dataset), td.cachers.Memory()))
    train_dataset.map(td.maps.To(T_train, 0))
                                
    eb_dataset = imdb.IMDB(root=f'{os.path.expanduser("~")}/data', 
                            train=False, 
                            split='EB1' if config.split == 'EB2' else 'EB2', 
                            target=config.target_attr)
    eb_dataset = td.datasets.WrapDataset(eb_dataset)
    eb_dataset.map(td.maps.To(T_test, 0))
    eb_dataset.cache(td.modifiers.UpToPercentage(cache_p, len(eb_dataset), td.cachers.Memory()))

    test_dataset = imdb.IMDB(root=f'{os.path.expanduser("~")}/data', 
                                train=False, 
                                split='test', 
                                target=config.target_attr)
    test_dataset = td.datasets.WrapDataset(test_dataset)
    test_dataset.map(td.maps.To(T_test, 0))
    test_dataset.cache(td.modifiers.UpToPercentage(cache_p, len(test_dataset), td.cachers.Memory()))

    # 4-Fold on test dataset
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    folds = []

    stratify = test_dataset.df[config.target_attr].copy()
    if config.target_attr == 'age':
        stratify = stratify.apply(imdb.bin_age) 

    for test_idx, valid_idx in kfold.split(test_dataset.df.image, stratify):
        folds.append((test_idx, valid_idx))

    fold = 0
    test_idx, valid_idx = folds[fold]
    print(f'Got {len(test_idx)} test images and {len(valid_idx)} valid images for fold {fold}')

    fold_valid_dataset = torch.utils.data.Subset(test_dataset, valid_idx)
    fold_test_dataset = torch.utils.data.Subset(test_dataset, test_idx)

    print('=> Train dataset =', len(train_dataset))
    print('=> Valid dataset =', len(fold_valid_dataset))
    print('=> EB test =', len(eb_dataset))
    print('=> Test =', len(fold_test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=shuffle,
                                               batch_size=config.batch_size,
                                               num_workers=8,
                                               persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(fold_valid_dataset,
                                               shuffle=False,
                                               batch_size=256,
                                               num_workers=4,
                                               persistent_workers=True)

    eb_test_loader = torch.utils.data.DataLoader(eb_dataset,
                                                 shuffle=False,
                                                 batch_size=256,
                                                 num_workers=4,
                                                 persistent_workers=True)  

    test_loader = torch.utils.data.DataLoader(fold_test_dataset,
                                              shuffle=False,
                                              batch_size=256,
                                              num_workers=4,
                                              persistent_workers=True)


    return train_loader, {'valid': valid_loader, 'eb_test': eb_test_loader, 'test': test_loader}

def topk_accuracy(outputs, labels, topk=1):
    outputs = torch.softmax(outputs, dim=1)
    _, preds = outputs.topk(topk, dim=1)
    
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds)).sum().float()
    return 100.*(correct / float(len(outputs))).cpu().item()

#def topk_accuracy(outputs, labels, topk=1):
#    _, predictions = torch.max(outputs, dim=1)
#    return (predictions == labels).sum().float() / float(len(labels))

def run(encoder, classifier, dataloader, criterion, optimizer, device):
    train = optimizer is not None

    tot_loss = 0.
    outputs = []
    targets = []
    f_outputs = []
    bias_targets = []

    classifier.train(train)
    for data, target, bias_target in tqdm(dataloader):
        data, target, bias_target = data.to(device), target.to(device), bias_target.to(device)
        
        with torch.no_grad():
            f_output, feats = encoder(data)

        with torch.set_grad_enabled(train):
            output = classifier(feats)
            loss = criterion(output, bias_target)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tot_loss += loss.item()
        outputs.append(output.detach().float())
        targets.append(target)
        bias_targets.append(bias_target)
        f_outputs.append(f_output.detach().float())

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    bias_targets = torch.cat(bias_targets, dim=0)
    f_outputs = torch.cat(f_outputs, dim=0)

    accs = {
        #'target': topk_accuracy(outputs, targets, topk=1),
        'bias': topk_accuracy(outputs, bias_targets, topk=1),
        'f_target': topk_accuracy(f_outputs, targets, topk=1)
    }

    return {'loss': tot_loss / len(dataloader), 'accuracy': accs}

def main(config):
    utils.set_seed(config.seed)

    device = torch.device('cuda')
    encoder = resnet18(n_classes=2 if config.target_attr == 'gender' else 12)

    checkpoint = torch.load(os.path.join('checkpoints', 'imdb', config.split,  config.crit, f'model{config.target_attr}.pth'), map_location='cpu')
    
    for key in list(checkpoint['model'].keys()):
        if 'm.' not in key:
            checkpoint['model'][f'm.{key}'] = checkpoint['model'].pop(key)
    
    encoder.load_state_dict(checkpoint['model'])
    encoder = encoder.to(device)
    encoder.eval()

    num_classes = 12
    if config.target_attr == 'age':
        num_classes = 2
    classifier = nn.Sequential(
        nn.Linear(512, num_classes)
    ).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loaders = load_imdb(config)

    wandb.init(
        project='rebias-classifiers',
        job_type='bias-classifier-imdb',
        config=config
    )

    print(config)

    for epoch in range(50):
        train_logs = run(encoder, classifier, train_loader, criterion, optimizer, device)
        print('Train:', train_logs)
        wandb.log({'train': train_logs}, commit=False)

        for key, val_loader in val_loaders.items():
            val_log = run(encoder, classifier, val_loader, criterion, None, device)
            print(key, val_log)
            wandb.log({key: val_log}, commit=False)
        
        wandb.log({'epoch': epoch+1})

        torch.save(classifier, os.path.join('checkpoints', 'imdb', config.split, config.crit, f'bias-classifier{config.target_attr}.pth'))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_attr', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['EB1', 'EB2'], required=True)
    parser.add_argument('--crit', type=str, choices=['vanilla', 'end'], required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    config = parser.parse_args()
    config.dataset = 'imdb'

    main(config)