#!/usr/bin/env python3

import argparse
import importlib.util
import sys
from pathlib import Path


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BiRefNet .pth weights to TorchScript for LibTorch inference.")
    parser.add_argument("--birefnet-dir", required=True, help="Path to a BiRefNet checkout")
    parser.add_argument("--weights", required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--output", required=True, help="Where to write the TorchScript file")
    parser.add_argument("--input-width", type=int, default=2048, help="Model input width")
    parser.add_argument("--input-height", type=int, default=2048, help="Model input height")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Tracing device")
    parser.add_argument("--mode", default="trace", choices=["trace", "script"], help="TorchScript export mode")
    parser.add_argument("--strict", action="store_true", help="Use strict tracing")
    args = parser.parse_args()

    birefnet_dir = Path(args.birefnet_dir).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not birefnet_dir.exists():
        raise FileNotFoundError(f"BiRefNet directory not found: {birefnet_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    sys.path.insert(0, str(birefnet_dir))

    import torch

    birefnet_module = load_module("birefnet_model", birefnet_dir / "models" / "birefnet.py")
    utils_module = load_module("birefnet_utils", birefnet_dir / "utils.py")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    model = birefnet_module.BiRefNet(bb_pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = utils_module.check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        if args.mode == "script":
            exported = torch.jit.script(model)
        else:
            dummy = torch.randn(1, 3, args.input_height, args.input_width, device=device)
            exported = torch.jit.trace(model, dummy, strict=args.strict)
        exported = torch.jit.freeze(exported.eval())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(output_path))
    print(f"Exported TorchScript model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
