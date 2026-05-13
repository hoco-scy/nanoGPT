"""
从 loss.json 绘制训练曲线。

用法:
    python plot_loss.py                          # 默认读取 out/loss.json
    python plot_loss.py --loss_file=out-shakespeare_char/loss.json
    python plot_loss.py --loss_file=out/loss.json --out_file=loss_curve.png
"""

import json
import argparse

import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，适合服务器/远程环境
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves from loss.json')
    parser.add_argument('--loss_file', type=str, default='out/loss.json', help='path to loss.json')
    parser.add_argument('--out_file', type=str, default=None, help='output image path (default: same dir as loss_file)')
    args = parser.parse_args()

    with open(args.loss_file, 'r') as f:
        data = json.load(f)

    train_data = data['train']
    val_data = data['val']

    fig, ax = plt.subplots(figsize=(10, 6))

    if train_data:
        train_iters = [d['iter'] for d in train_data]
        train_losses = [d['loss'] for d in train_data]
        ax.plot(train_iters, train_losses, label='train loss', linewidth=1.5)

    if val_data:
        val_iters = [d['iter'] for d in val_data]
        val_losses = [d['loss'] for d in val_data]
        ax.plot(val_iters, val_losses, label='val loss', linewidth=1.5, marker='o', markersize=4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_file = args.out_file
    if out_file is None:
        import os
        out_file = os.path.join(os.path.dirname(args.loss_file) or '.', 'loss_curve.png')

    fig.savefig(out_file, dpi=150)
    print(f"loss curve saved to {out_file}")


if __name__ == '__main__':
    main()
