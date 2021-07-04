from datasets.colour_mnist import get_biased_mnist_dataloader
from models.mnist_models import SimpleConvNet
import torch
import torch.nn as nn
import torchvision
import argparse
import os

import wandb
from tqdm import tqdm

def topk_accuracy(outputs, labels, topk=1):
    outputs = torch.softmax(outputs, dim=1)
    _, preds = outputs.topk(topk, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds)).sum()
    return 100.*(correct / float(len(outputs))).cpu().item()

def run(encoder, classifier, dataloader, criterion, optimizer, device):
    train = optimizer is not None

    tot_loss = 0.
    outputs = []
    targets = []

    classifier.train(train)
    for data, _, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            _, feats = encoder(data)

        with torch.set_grad_enabled(train):
            output = classifier(feats)
            loss = criterion(output, target)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tot_loss += loss.item()
        outputs.append(output.detach().float())
        targets.append(target)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    accs = {
        'top1': topk_accuracy(outputs, targets, topk=1),
    }

    return {'loss': tot_loss / len(dataloader), 'accuracy': accs}

def main(config):
    device = torch.device('cuda')
    checkpoint = torch.load(os.path.join('checkpoints', config.crit, f'best{config.rho}.pth'), map_location='cpu')
    
    encoder = SimpleConvNet({'kernel_size': 7, 'feature_pos': 'post'})
    for key in list(checkpoint['f_net'].keys()):
        checkpoint['f_net'][key.replace('module.', '')] = checkpoint['f_net'].pop(key)

    encoder.load_state_dict(checkpoint['f_net'])
    encoder = encoder.to(device)
    encoder.eval()

    classifier = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    batch_size = 256
    tr_loader = get_biased_mnist_dataloader(config.root, batch_size=batch_size,
                                            data_label_correlation=0.999,
                                            n_confusing_labels=9,
                                            train=True)
    val_loaders = {}
    val_loaders['biased'] = get_biased_mnist_dataloader(config.root, batch_size=batch_size,
                                                        data_label_correlation=1,
                                                        n_confusing_labels=9,
                                                        train=False)
                                                        
    val_loaders['rho0'] = get_biased_mnist_dataloader(config.root, batch_size=batch_size,
                                                      data_label_correlation=0,
                                                      n_confusing_labels=9,
                                                      train=False)

    val_loaders['unbiased'] = get_biased_mnist_dataloader(config.root, batch_size=batch_size,
                                                          data_label_correlation=0.1,
                                                          n_confusing_labels=9,
                                                          train=False)

    wandb.init(
        project='rebias-classifiers',
        job_type='bias-classifier',
        config=config
    )

    print(config)

    for epoch in range(50):
        train_logs = run(encoder, classifier, tr_loader, criterion, optimizer, device)
        wandb.log({'train': train_logs}, commit=False)
        print('Train:', train_logs)

        for key, val_loader in val_loaders.items():
            val_log = run(encoder, classifier, val_loader, criterion, None, device)
            wandb.log({key: val_log}, commit=False)
            print(key, val_log)
        
        wandb.log({'epoch': epoch+1})

        torch.save(classifier, os.path.join('checkpoints', config.crit, f'bias-classifier{config.rho}.pth'))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=f'{os.path.expanduser("~")}/data/MNIST')
    parser.add_argument('--rho', type=float, required=True)
    parser.add_argument('--crit', type=str, choices=['vanilla', 'rubi', 'rebias', 'learned-mixin'], required=True)
    config = parser.parse_args()

    main(config)