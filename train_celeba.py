import torch
import torch.nn as nn
from torch.serialization import load
import torchvision
from datasets import celeba
import numpy as np
import argparse
import os

import utils
import wandb
from tqdm import tqdm

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


def load_celeba(config, shuffle=True):
    bias_attr = 'Male'

    train_dataset = celeba.CelebA(root=f'{os.path.expanduser("~")}/data', split='train', target=config.target_attr, bias_attr=bias_attr, seed=config.seed)
    valid_dataset = celeba.CelebA(root=f'{os.path.expanduser("~")}/data', split='train', target=config.target_attr, bias_attr=bias_attr, seed=config.seed)
    unbiased_dataset = celeba.CelebA(root=f'{os.path.expanduser("~")}/data', split='valid', unbiased=True, target=config.target_attr, bias_attr=bias_attr, seed=config.seed)
    conflicting_dataset = celeba.CelebA(root=f'{os.path.expanduser("~")}/data', split='valid', unbiased=False, target=config.target_attr, bias_attr=bias_attr, seed=config.seed)

    target = config.target_attr
    min_size = int(0.2*train_dataset.attr_df.groupby([target, bias_attr]).count().min()['image'])
    valid_dataset.attr_df = valid_dataset.attr_df.groupby([target, bias_attr]).apply(lambda group: group.sample(min_size, random_state=config.seed)).reset_index(drop=True).copy()
    train_dataset.attr_df = train_dataset.attr_df[~train_dataset.attr_df.image.isin(valid_dataset.attr_df.image)].copy()
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        num_workers=8,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True
    )

    unbiased_loader = torch.utils.data.DataLoader(
        unbiased_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True
    )

    conflicting_loader = torch.utils.data.DataLoader(
        conflicting_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, {'valid': valid_loader, 'unbiased_test': unbiased_loader, 'bias_conflicting_test': conflicting_loader}

"""def topk_accuracy(outputs, labels, topk=1):
    outputs = torch.softmax(outputs, dim=1)
    _, preds = outputs.topk(topk, dim=1)
    
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds)).sum()
    return 100.*(correct / float(len(outputs))).cpu().item()"""

def topk_accuracy(outputs, labels, topk=1):
    _, predictions = torch.max(outputs, dim=1)
    return (predictions == labels).sum().float() / float(len(labels))

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
        'target': topk_accuracy(outputs, targets, topk=1),
        'bias': topk_accuracy(outputs, bias_targets, topk=1),
        'f_target': topk_accuracy(f_outputs, targets, topk=1)
    }

    return {'loss': tot_loss / len(dataloader), 'accuracy': accs}

def main(config):
    utils.set_seed(config.seed)

    device = torch.device('cuda')
    encoder = resnet18(n_classes=2)

    #checkpoint = torch.load(os.path.join('checkpoints', 'celeba', config.crit, f'model{config.target_attr}.pth'), map_location='cpu')
    #encoder.load_state_dict(checkpoint['model'])
    encoder = encoder.to(device)
    encoder.eval()

    classifier = nn.Sequential(
        nn.Linear(512, 2)
    ).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    batch_size = 256
    train_loader, val_loaders = load_celeba(config)

    wandb.init(
        project='rebias-classifiers',
        job_type='bias-classifier-celeba',
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

        torch.save(classifier, os.path.join('checkpoints', 'celeba', config.crit, f'bias-classifier{config.target_attr}.pth'))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_attr', type=str, required=True)
    parser.add_argument('--crit', type=str, choices=['vanilla', 'end'], required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    config = parser.parse_args()

    main(config)