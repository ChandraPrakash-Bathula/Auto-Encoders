"""Evaluate a trained Autoencoder: metrics, figures and latent-code export.

    python -m autoencoder.evaluate --run-dir outputs/ae/z128

Writes (inside the run dir): metrics_test.json, figures/{reconstructions,latent_perturbation,
latent_interpolation}_test.png and latents/<split>.npz (input for analysis/visualize_latents.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from autoencoder.model import Autoencoder
from common.dataset import SliceDataset
from common.evaluation import (GaussianLatentSampler, encode_dataset, evaluate_split, interpolation_figure,
                               perturbation_figure, realism_metrics, reconstruction_figure, save_latents,
                               summarize)
from common.utils import count_parameters, get_device, save_image_grid, save_json, seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the Autoencoder")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--data-dir", default=None, help="override the data dir stored in the checkpoint")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--latent-splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--max-latents", type=int, default=6000, help="max points saved per split")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-fid", type=int, default=10000, help="max real/generated images for FID")
    p.add_argument("--no-fid", action="store_true", help="skip FID / generation metrics (debug)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ckpt = Autoencoder.from_checkpoint(args.run_dir / "checkpoints" / args.checkpoint, device)
    cfg = ckpt["train_config"]
    seed_everything(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    fig_dir, lat_dir = args.run_dir / "figures", args.run_dir / "latents"
    fig_dir.mkdir(exist_ok=True); lat_dir.mkdir(exist_ok=True)

    def forward_fn(x):
        z = model.encode(x)
        return model.decode(z), z, None

    results = {}
    for split in sorted(set([args.split] + args.latent_splits)):
        ds = SliceDataset(data_dir, split)
        per, latents, _ = evaluate_split(ds, forward_fn, device, args.batch_size, args.num_workers)
        results[split] = (ds, per, latents)
        if split in args.latent_splits:
            save_latents(lat_dir / f"{split}.npz", ds, latents, per, max_points=args.max_latents)

    ds, per, latents = results[args.split]
    metrics = {"model": "ae", "run_name": args.run_dir.name, "latent_dim": cfg["latent_dim"], "beta": None,
               "split": args.split, "checkpoint": args.checkpoint, "epoch": ckpt["epoch"],
               "n_images": len(ds), "n_subjects": int(len(set(ds.subject_ids.tolist()))),
               "n_params": count_parameters(model), **summarize(per)}

    # Generation for a deterministic AE: sample a full-covariance Gaussian fitted to training codes.
    gauss = GaussianLatentSampler(encode_dataset(SliceDataset(data_dir, "train"), model.encode, device))
    gen_gauss = lambda k: model.decode(gauss(k))
    if not args.no_fid:
        metrics.update(realism_metrics(ds, data_dir, args.split, device, model.reconstruct,
                                       {"gauss": gen_gauss}, n_fid=args.n_fid))
    save_json(metrics, args.run_dir / f"metrics_{args.split}.json")

    reconstruction_figure(lambda x: model.reconstruct(x), ds, device,
                          fig_dir / f"reconstructions_{args.split}.png", "AE recon")
    perturbation_figure(model.encode, model.decode, ds, latents, device,
                        fig_dir / f"latent_perturbation_{args.split}.png")
    interpolation_figure(model.interpolate, ds, device, fig_dir / f"latent_interpolation_{args.split}.png")
    torch.manual_seed(0)
    save_image_grid(gen_gauss(64), fig_dir / "generated_samples_gaussfit.png", n_cols=8,
                    title=f"AE samples, z ~ Gaussian fit to training codes  (z={cfg['latent_dim']})")

    print(f"[AE z={cfg['latent_dim']}] {args.split}: MSE {metrics['mse_mean']:.5f} | "
          f"PSNR {metrics['psnr_mean']:.2f} dB | SSIM {metrics['ssim_mean']:.4f}"
          + (f" | FID recon {metrics['fid_recon']:.1f} | FID gauss {metrics['fid_gauss']:.1f} "
             f"(real floor {metrics['fid_real_floor']:.1f})" if "fid_recon" in metrics else ""))


if __name__ == "__main__":
    main()
