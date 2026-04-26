# BiRefNet Integration Notes

这个项目里，C++ 侧不能直接消费上游的 `.pth`。

要做本地 C++ 推理，推荐路线是：

1. 用 Python 侧把 BiRefNet 权重导出成 `.onnx`
2. 在 Nuke 插件里用 ONNX Runtime C++ 加载 `.onnx`
3. 做整帧预处理 / 推理 / 后处理 / 缓存

## You Need From Upstream

必需文件：

- BiRefNet 仓库本身
- 目标权重文件 `.pth`
- `deform_conv2d_onnx_exporter.py`

原因：

- 上游模型包含 `torchvision.ops.deform_conv2d`
- 作者的 ONNX notebook 里专门处理了这个导出点
- 所以不能把 `.pth` 当成一个“普通网络”直接随便 export

## Recommended Inputs

优先建议你直接用这些现成模型之一：

- `BiRefNet-matting`
- `BiRefNet_HR-matting`
- `BiRefNet_dynamic`

如果你的目标是“精细化抠图”，优先从 matting 系列开始。

## Export Flow

导出脚本：

- [tools/export_birefnet_onnx.py](/Users/jiadaixi/Desktop/matte-nuke-plugin/tools/export_birefnet_onnx.py:1)

示例：

```bash
python3 tools/export_birefnet_onnx.py \
  --birefnet-dir /path/to/BiRefNet \
  --weights /path/to/BiRefNet-matting.pth \
  --output /path/to/models/BiRefNet-matting.onnx \
  --input-width 1024 \
  --input-height 1024 \
  --device cuda \
  --deform-exporter /path/to/deform_conv2d_onnx_exporter.py
```

## C++ Runtime Expectations

当前 C++ 后端按上游 notebook 的默认假设实现：

- 输入：RGB float，按 ImageNet mean/std 归一化
- 输入 tensor：`1x3xH xW`，NCHW
- 输出：单通道 logits
- 后处理：`sigmoid` 后 resize 回原始分辨率

## What Is Still Missing In The Nuke Node

当前节点骨架还没有把“整帧推理缓存”接完。

这一步必须做，因为 BiRefNet 是整图网络，不适合按 scanline 一行一行单独跑。合理做法是：

1. 在一次 render 请求里抓整帧 RGBA
2. 转成 `ImageTensor`
3. 调 `BiRefNetOnnxBackend::infer(...)`
4. 把返回的整张 alpha mask 缓存起来
5. 在 `engine(...)` 里按当前行读缓存结果

## Practical Recommendation

如果你想尽快验证整条链路，最稳的顺序是：

1. 先拿作者已经发布的 `.onnx`
2. 先单独写一个 CLI 小程序验证 C++ 推理
3. 再把这套推理结果接回 Nuke 节点

这样能把“模型问题”和“Nuke 采样/缓存问题”拆开。 
