import torch
import torch.nn.functional as F
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from network import UNet
from datasets import Neighbour2Neighbour
from transforms import generate_mask_pair, generate_subimages


def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_patch_size(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError('patch_size must be int or a list/tuple of length 2')


def train():
    parser = argparse.ArgumentParser(description='Neighbour2Neighbour training')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = _load_config(args.config)

    model = UNet(
        in_nc=cfg.get('in_channels', 1),
        out_nc=cfg.get('out_channels', 1),
        n_feature=cfg.get('n_feature', 48),
        blindspot=cfg.get('blindspot', False),
        zero_last=cfg.get('zero_last', False),
    )

    device = torch.device(
        cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu'
    )
    model.to(device)

    epochs = cfg.get('epochs', 100)
    batch_size = cfg.get('batch_size', 8)
    patch_size = _parse_patch_size(cfg.get('patch_size', 512))
    dataset_path = cfg.get('dataset_path')
    if not dataset_path:
        raise ValueError('dataset_path must be set in config.yaml')

    lambda1 = float(cfg.get('lambda1', 1.0))
    lambda2 = float(cfg.get('lambda2', 1.0))
    increase_ratio = float(cfg.get('increase_ratio', 0.5))

    lr = cfg.get('lr', 1e-3)
    scheduler_tmax = cfg.get('scheduler_tmax', 20)
    scheduler_eta_min = cfg.get('scheduler_eta_min', 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_tmax, eta_min=scheduler_eta_min
    )

    full_dataset = Neighbour2Neighbour(dataset_path, patch_size=patch_size)
    val_split = float(cfg.get('val_split', 0.1))
    val_len = int(len(full_dataset) * val_split)
    train_len = len(full_dataset) - val_len
    if val_len > 0:
        generator = torch.Generator().manual_seed(cfg.get('split_seed', 42))
        train_dataset, val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )
    else:
        train_dataset, val_dataset = full_dataset, full_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg.get('num_workers', 4),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.get('val_batch_size', 1), shuffle=False,
        num_workers=cfg.get('val_num_workers', 0),
    )

    print(f'train samples: {len(train_dataset)}, val samples: {len(val_dataset)}')

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ===== train =====
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        for source in loop:
            source = source.to(device)                     # (B, C, H, W)

            # 每个 batch 随机生成一次 mask，对 batch 内所有图像统一处理
            mask1, mask2 = generate_mask_pair(source)
            noisy_sub1 = generate_subimages(source, mask1)
            noisy_sub2 = generate_subimages(source, mask2)

            # 两次预测
            noisy_output = model(noisy_sub1)
            noisy_target = noisy_sub2

            # 一致性正则化
            with torch.no_grad():
                noisy_denoised = model(source)
                sub1_denoised = generate_subimages(noisy_denoised, mask1)
                sub2_denoised = generate_subimages(noisy_denoised, mask2)

            Lambda = epoch / epochs * increase_ratio

            diff = noisy_output - noisy_target
            exp_diff = sub1_denoised - sub2_denoised

            loss1 = torch.mean(diff ** 2)
            loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
            loss = lambda1 * loss1 + lambda2 * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')

        # ===== validation =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for source in tqdm(val_loader, total=len(val_loader), leave=False,
                               desc='validation'):
                source = source.to(device)

                mask1, mask2 = generate_mask_pair(source)
                noisy_sub1 = generate_subimages(source, mask1)
                noisy_sub2 = generate_subimages(source, mask2)

                output = model(noisy_sub1)
                loss = F.mse_loss(output, noisy_sub2)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'Epoch: {epoch + 1}\t train_loss: {train_loss:.6f}\t '
              f'val_loss: {val_loss:.6f}')

        scheduler.step()

        if val_loss < best_val_loss:
            print(f'val_loss improved: {best_val_loss:.6f} -> {val_loss:.6f}  saving...')
            best_val_loss = val_loss
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), './checkpoints/weight.pth')

    # ===== save history =====
    os.makedirs('./results', exist_ok=True)
    with open('./results/train_loss.txt', 'w') as f:
        for v in train_loss_history:
            f.write(f'{v}\n')
    with open('./results/val_loss.txt', 'w') as f:
        for v in val_loss_history:
            f.write(f'{v}\n')

    print('Finished')


if __name__ == '__main__':
    train()
