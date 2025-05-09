import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
from dataset import SPMIDataset, get_transforms
from model import LeNet, WideResNet
from train import train


def parse_args():
    
    # dEFAULT : python main.py --dataset cifar10 --num_labeled 4000 --partial_rate 0.3...... basicically the second for for cifar10
    #Experiments that we need to run:
    # Fmisnt - python main.py --dataset fashion_mnist --num_labeled 1000 --partial_rate 0.3
               #python main.py --dataset fashion_mnist --num_labeled 4000 --partial_rate 0.3
    # CIFAR10 - python main.py --dataset cifar10 --num_labeled 1000 --partial_rate 0.3
                #python main.py --dataset cifar10 --num_labeled 4000 --partial_rate 0.3 :)
                #python main.py --dataset cifar10 --num_labeled 4000 --partial_rate 0.7
    # CIFAR100 - python main.py --dataset cifar100 --num_labeled 5000 --partial_rate 0.05
                #python main.py --dataset cifar100 --num_labeled 10000 --partial_rate 0.05
    # SVHN - python main.py --dataset svhn --num_labeled 1000 --partial_rate 0.3
    parser = argparse.ArgumentParser(description='SPMI method implementation')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['fashion_mnist', 'cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_labeled', type=int, default=4000)
    parser.add_argument('--partial_rate', type=float, default=0.3)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # SPMI settings
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--unlabeled_tau', type=float, default=2.0)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--strong_aug', action='store_true', default=True)
    
    # seeding, gpu etc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # parse args
    args = parse_args()
    set_seed(args.seed)
    #device set
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # get transforms
    transform = get_transforms(args.dataset, strong_aug=args.strong_aug)
    
    # creaate dataset
    dataset = SPMIDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        num_labeled=args.num_labeled,
        partial_rate=args.partial_rate,
        transform=transform,
        download=True,
        seed=args.seed
    )
    
    # setup dataloader inst
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # setup models
    if args.dataset == 'fashion_mnist':
        model = LeNet(num_classes=10)
    elif args.dataset == 'cifar100':
        model = WideResNet(depth=28, widen_factor=8, num_classes=100)
    else:  # cifar10 or svhn
        model = WideResNet(depth=28, widen_factor=2, num_classes=10)
    
    model = model.to(device)
    print(f"Created {type(model).__name__} model")
    
    # SGD for optim
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    # train
    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        tau=args.tau,
        unlabeled_tau=args.unlabeled_tau,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        eval_interval=args.eval_interval
    )


if __name__ == "__main__":
    main()