import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from network import UNet


def _load_config(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _load_image(path):
    img = Image.open(path)
    if img.mode in ('I;16', 'I;16B', 'I;16L'):
        arr = np.array(img, dtype=np.uint16).astype(np.float32)
        max_val = 65535.0
    else:
        arr = np.array(img.convert('L'), dtype=np.uint8).astype(np.float32)
        max_val = 255.0

    if arr.ndim == 2:
        arr = arr[:, :, None]

    arr = arr / max_val
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, max_val


def _save_image(arr, path, max_val):
    arr = np.clip(arr * max_val, 0, max_val).round()
    if max_val > 255:
        out = arr.astype(np.uint16)
        Image.fromarray(out, mode='I;16').save(path)
    else:
        out = arr.astype(np.uint8)
        Image.fromarray(out, mode='L').save(path)


def main():
    parser = argparse.ArgumentParser(description='Noise2Void inference')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = _load_config(args.config)

    weights_path = cfg.get('weights_path', './checkpoints/weight.pth')
    input_dir = cfg.get('input_dir')
    output_dir = cfg.get('output_dir', './outputs')
    if not input_dir:
        raise ValueError('input_dir must be set in config.yaml')

    in_channels = cfg.get('in_channels', 1)
    out_channels = cfg.get('out_channels', 1)
    depth = cfg.get('depth', 5)
    padding = cfg.get('padding', True)
    batch_norm = cfg.get('batch_norm', True)
    up_mode = cfg.get('up_mode', 'upconv')
    interpolation_mode = cfg.get('interpolation_mode', 'nearest')

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        padding=padding,
        batch_norm=batch_norm,
        up_mode=up_mode,
        interpolation_mode=interpolation_mode,
    )

    if torch.cuda.is_available() and cfg.get('device', '').startswith('cuda'):
        device = torch.device(cfg.get('device', 'cuda:0'))
    else:
        device = torch.device('cpu')

    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    paths = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]

    if not paths:
        raise RuntimeError(f'No images found in {input_dir}')

    with torch.no_grad():
        for path in paths:
            tensor, max_val = _load_image(path)
            tensor = tensor.to(device)
            pred = model(tensor)
            # pred = torch.clamp(pred, 0.0, 1.0)
            print(
                '{}: input=({:.4f},{:.4f}) pred=({:.4f},{:.4f})'.format(
                    path.name,
                    tensor.min().item(), tensor.max().item(),
                    pred.min().item(), pred.max().item(),
                )
            )
            pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if pred.shape[2] == 1:
                pred = pred[:, :, 0]

            out_path = output_dir / f'{path.stem}_denoised.png'
            _save_image(pred, out_path, max_val)

    print(f'Done. Saved to {output_dir}')


if __name__ == '__main__':
    main()
