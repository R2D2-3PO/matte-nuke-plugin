# BiRefNet Integration Notes

这个项目里，C++ 侧不能直接消费上游的 `.pth`。

要做本地 C++ 推理，推荐路线是：

1. 用 Python 侧把 BiRefNet 权重导出成 TorchScript
2. 在 Nuke 插件里用 LibTorch C++ 加载 TorchScript
3. 做整帧预处理 / 推理 / 后处理 / 缓存

## You Need From Upstream

必需文件：

- BiRefNet 仓库本身
- 目标权重文件 `.pth`
- `torchvision/_C.so`

原因：

- 上游模型包含 `torchvision.ops.deform_conv2d`
- TorchScript 导出可以保留这个 TorchVision 自定义算子
- 但 C++ 运行时也要把 TorchVision 的原生算子库加载进来

## Recommended Inputs

优先建议你直接用这些现成模型之一：

- `BiRefNet-matting`
- `BiRefNet_HR-matting`
- `BiRefNet_dynamic`

如果你的目标是“精细化抠图”，优先从 matting 系列开始。

## Export Flow

导出脚本：

- [tools/export_birefnet_torchscript.py](/Users/jiadaixi/Desktop/matte-nuke-plugin/tools/export_birefnet_torchscript.py:1)

示例：

```bash
python3 tools/export_birefnet_torchscript.py \
  --birefnet-dir /path/to/BiRefNet \
  --weights /path/to/BiRefNet-matting.pth \
  --output /path/to/models/BiRefNet-matting.ts \
  --mode script \
  --input-width 1024 \
  --input-height 1024 \
  --device cuda
```

如果你的模型改动里引入了明显的控制流，优先试 `--mode script`。如果结构是静态的，`trace` 往往更直接。

## C++ Runtime Expectations

当前 C++ 后端按上游推理逻辑实现：

- 输入：RGB float，按 ImageNet mean/std 归一化
- 输入 tensor：`1x3xH xW`，NCHW
- 输出：从 TorchScript 返回值里递归提取最后一个预测 tensor
- 后处理：`sigmoid` 后 resize 回原始分辨率

## What Is Still Missing In The Nuke Node

当前节点骨架还没有把“整帧推理缓存”接完。

这一步必须做，因为 BiRefNet 是整图网络，不适合按 scanline 一行一行单独跑。合理做法是：

1. 在一次 render 请求里抓整帧 RGBA
2. 转成 `ImageTensor`
3. 调 `BiRefNetTorchBackend::infer(...)`
4. 把返回的整张 alpha mask 缓存起来
5. 在 `engine(...)` 里按当前行读缓存结果

## Practical Recommendation

如果你想尽快验证整条链路，最稳的顺序是：

1. 先导出一个 TorchScript 模型
2. 确认 `torchvision/_C.so` 路径可用
3. 再把这套推理结果接回 Nuke 节点

这样能把“模型问题”和“Nuke 采样/缓存问题”拆开。 
