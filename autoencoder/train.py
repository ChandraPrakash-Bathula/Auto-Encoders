"""Train the Autoencoder.

    python -m autoencoder.train                              # defaults from autoencoder/config.yaml
    python -m autoencoder.train --latent-dim 16 --epochs 60
    python -m autoencoder.train --resume outputs/ae/z128/checkpoints/last.pt

Outputs -> <output_dir>/ae/z<latent_dim>/{checkpoints,figures,history.csv,config_resolved.yaml}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from autoencoder.model import AEConfig, Autoencoder, model_config_dict
from common.augment import augment_batch
from common.dataset import make_loaders
from common.metrics import reconstruction_metrics
from common.utils import (AverageMeter, count_parameters, get_device, load_config,
                          plot_history, save_comparison_grid, save_json, save_yaml,
                          seed_everything, write_history_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the convolutional Autoencoder")
    p.add_argument("--config", default="autoencoder/config.yaml")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--latent-dim", dest="latent_dim", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--weight-decay", dest="weight_decay", type=float)
    p.add_argument("--patience", type=int, help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--augment", type=int, choices=[0, 1], help="train-time augmentation on (1) / off (0)")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None, help="path to last.pt to continue training")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    meters = {k: AverageMeter() for k in ("loss", "mse", "psnr", "ssim")}
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            recon, _ = model(x)
        recon = recon.float()
        m = reconstruction_metrics(x, recon)
        n = x.shape[0]
        meters["loss"].update(Autoencoder.loss_function(recon, x).item(), n)
        for k in ("mse", "psnr", "ssim"):
            meters[k].update(m[k].mean().item(), n)
    return {k: v.avg for k, v in meters.items()}


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "device", "resume")}
    cfg = load_config(args.config, overrides)
    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    device = get_device(args.device)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

    run_dir = Path(cfg["output_dir"]) / "ae" / f"z{cfg['latent_dim']}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    train_loader, val_loader, _ = make_loaders(cfg["data_dir"], cfg["batch_size"], cfg["num_workers"],
                                               pin_memory=device.type == "cuda")
    cfg["image_size"] = train_loader.dataset.image_size  # trust the data
    cfg["in_channels"] = train_loader.dataset.channels
    save_yaml(cfg, run_dir / "config_resolved.yaml")

    model_cfg = AEConfig(image_size=cfg["image_size"], in_channels=cfg["in_channels"],
                         latent_dim=cfg["latent_dim"], base_channels=cfg["base_channels"],
                         channel_mults=list(cfg["channel_mults"]))
    model = Autoencoder(model_cfg).to(device)
    print(f"Autoencoder | latent_dim={cfg['latent_dim']} | params={count_parameters(model)/1e6:.2f}M "
          f"| device={device} | train/val slices={len(train_loader.dataset)}/{len(val_loader.dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"],
                                                           eta_min=cfg["lr"] * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_val, start_epoch, bad_epochs = [], float("inf"), 1, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history, best_val = ckpt["history"], ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    vis_idx = torch.linspace(0, len(val_loader.dataset) - 1, cfg["num_vis"]).long()
    vis_batch = torch.stack([val_loader.dataset[int(i)] for i in vis_idx]).to(device)

    def save_ckpt(path: Path, epoch: int, val_metrics: dict) -> None:
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "model_config": model_config_dict(model_cfg),
                    "train_config": cfg, "epoch": epoch, "best_val": best_val,
                    "val_metrics": val_metrics, "history": history, "model_type": "ae"}, path)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        meter = AverageMeter()
        for x in tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}", leave=False):
            x = augment_batch(x.to(device, non_blocking=True), cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                recon, _ = model(x)
            loss = Autoencoder.loss_function(recon, x)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item(), x.shape[0])
        scheduler.step()

        val = validate(model, val_loader, device, use_amp)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": meter.avg,
               "val_loss": val["loss"], "val_psnr": val["psnr"], "val_ssim": val["ssim"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:3d}] train {row['train_loss']:.5f} | val {row['val_loss']:.5f} | "
              f"PSNR {row['val_psnr']:.2f} dB | SSIM {row['val_ssim']:.4f} | {row['seconds']}s")

        improved = val["loss"] < best_val
        if improved:
            best_val, bad_epochs = val["loss"], 0
            save_ckpt(run_dir / "checkpoints" / "best.pt", epoch, val)
        else:
            bad_epochs += 1
        save_ckpt(run_dir / "checkpoints" / "last.pt", epoch, val)

        write_history_csv(history, run_dir / "history.csv")
        if epoch % cfg["vis_every"] == 0 or epoch == cfg["epochs"] or epoch == start_epoch:
            model.eval()
            recon = model.reconstruct(vis_batch)
            save_comparison_grid([vis_batch, recon, (vis_batch - recon).abs().mean(1, keepdim=True) * 4],
                                 ["input", "AE recon", "|error| x4"],
                                 run_dir / "figures" / f"recon_epoch{epoch:03d}.png")
            plot_history(history, ["loss", "psnr", "ssim"], run_dir / "figures" / "training_curves.png",
                         title=f"Autoencoder (z={cfg['latent_dim']})")
        if cfg["patience"] and bad_epochs >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    save_json({"best_val_loss": best_val, "epochs_run": len(history)}, run_dir / "train_summary.json")
    print(f"Done. Best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")


if __name__ == "__main__":
    main()
