#!/usr/bin/env python3

import argparse
import importlib.util
import sys
from pathlib import Path


PATCH_OLD = "return sym_help._get_tensor_dim_size(tensor, dim)"
PATCH_NEW = """
tensor_dim_size = sym_help._get_tensor_dim_size(tensor, dim)
if tensor_dim_size == None and (dim == 2 or dim == 3):
    import typing
    from torch import _C

    x_type = typing.cast(_C.TensorType, tensor.type())
    x_strides = x_type.strides()

    tensor_dim_size = x_strides[2] if dim == 3 else x_strides[1] // x_strides[2]
elif tensor_dim_size == None and (dim == 0):
    import typing
    from torch import _C

    x_type = typing.cast(_C.TensorType, tensor.type())
    x_strides = x_type.strides()
    tensor_dim_size = x_strides[3]

return tensor_dim_size
""".strip()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_deform_exporter(path: Path) -> None:
    content = path.read_text()
    if PATCH_NEW in content:
        return
    if PATCH_OLD not in content:
        raise RuntimeError(
            "Unexpected deform_conv2d_onnx_exporter.py content. "
            "Please compare it with the patch in BiRefNet/tutorials/BiRefNet_pth2onnx.ipynb."
        )
    path.write_text(content.replace(PATCH_OLD, PATCH_NEW))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BiRefNet .pth weights to ONNX for C++/ONNX Runtime inference.")
    parser.add_argument("--birefnet-dir", required=True, help="Path to a BiRefNet checkout")
    parser.add_argument("--weights", required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--output", required=True, help="Where to write the .onnx file")
    parser.add_argument("--input-width", type=int, default=1024, help="Model input width")
    parser.add_argument("--input-height", type=int, default=1024, help="Model input height")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device")
    parser.add_argument(
        "--deform-exporter",
        default="deform_conv2d_onnx_exporter.py",
        help="Path to deform_conv2d_onnx_exporter.py from masamitsu-murase/deform_conv2d_onnx_exporter",
    )
    args = parser.parse_args()

    birefnet_dir = Path(args.birefnet_dir).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    deform_exporter_path = Path(args.deform_exporter).expanduser().resolve()

    if not birefnet_dir.exists():
        raise FileNotFoundError(f"BiRefNet directory not found: {birefnet_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not deform_exporter_path.exists():
        raise FileNotFoundError(
            "deform_conv2d_onnx_exporter.py not found. "
            "Fetch it from https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter"
        )

    sys.path.insert(0, str(birefnet_dir))

    import torch
    from torchvision.ops.deform_conv import DeformConv2d  # noqa: F401

    patch_deform_exporter(deform_exporter_path)
    deform_exporter = load_module("deform_conv2d_onnx_exporter", deform_exporter_path)
    deform_exporter.register_deform_conv2d_onnx_op()

    birefnet_module = load_module("birefnet_model", birefnet_dir / "models" / "birefnet.py")
    utils_module = load_module("birefnet_utils", birefnet_dir / "utils.py")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    model = birefnet_module.BiRefNet(bb_pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = utils_module.check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    dummy = torch.randn(1, 3, args.input_height, args.input_width, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        verbose=False,
        opset_version=args.opset,
        input_names=["input_image"],
        output_names=["output_image"],
    )

    print(f"Exported ONNX model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
