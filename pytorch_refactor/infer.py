import argparse
import torch
from PIL import Image
from torchvision import transforms

try:
    from .model import SwinTransformerV2MoE, SimMIM  # type: ignore
except ImportError:  # pragma: no cover
    from model import SwinTransformerV2MoE, SimMIM


def get_args():
    p = argparse.ArgumentParser("RingMoE SimMIM inference (reconstruction)")
    p.add_argument("--image", type=str, required=True, help="path to an input image")
    p.add_argument("--ckpt", type=str, required=True, help="path to a consolidated .pt (state_dict) checkpoint")
    p.add_argument("--out", type=str, default="recon.png", help="output path")
    p.add_argument("--mask_ratio", type=float, default=0.6)
    p.add_argument("--moe_experts", type=int, default=8)
    p.add_argument("--disable_moe", action="store_true", help="disable MoE (use when checkpoint was trained without DeepSpeed/MoE)")
    p.add_argument("--input_size", type=int, default=192)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_model(moe_experts: int, *, input_size: int, disable_moe: bool) -> SimMIM:
    moe_config = None if disable_moe else {"moe_stages": [2, 3], "num_experts": moe_experts}
    encoder = SwinTransformerV2MoE(
        img_size=input_size,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        moe_config=moe_config,
    )
    return SimMIM(encoder=encoder)


def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(args.moe_experts, input_size=args.input_size, disable_moe=bool(args.disable_moe))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    # random mask at patch level: [B, H_p, W_p]
    patch_size = int(getattr(model.encoder.patch_embed, "patch_size", (4, 4))[0])
    H_p = W_p = args.input_size // patch_size
    token_count = H_p * W_p
    mask_count = int(token_count * args.mask_ratio)
    idx = torch.randperm(token_count, device=device)[:mask_count]
    mask = torch.zeros(token_count, device=device, dtype=torch.int64)
    mask[idx] = 1
    mask = mask.view(1, H_p, W_p)

    with torch.no_grad():
        loss, x_rec, aux = model(x, mask)

    # denorm + save
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_rec = (x_rec * std + mean).clamp(0, 1)

    out = (x_rec[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(out).save(args.out)

    print(f"saved: {args.out}")
    print(f"loss={float(loss.detach().cpu()):.6f} aux={float(aux.detach().cpu()):.6f}")
    if missing:
        print(f"missing keys: {len(missing)}")
    if unexpected:
        print(f"unexpected keys: {len(unexpected)}")


if __name__ == "__main__":
    main()

