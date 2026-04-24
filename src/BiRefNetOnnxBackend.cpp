#include "BiRefNetPlugin/BiRefNetOnnxBackend.h"

#include <algorithm>
#include <utility>

namespace matte {

struct BiRefNetOnnxBackend::Impl {
    bool initialized = false;
    InferenceOptions options;
};

BiRefNetOnnxBackend::BiRefNetOnnxBackend() : impl_(std::make_unique<Impl>()) {}

BiRefNetOnnxBackend::~BiRefNetOnnxBackend() = default;

bool BiRefNetOnnxBackend::initialize(const InferenceOptions& options, std::string* errorMessage) {
    impl_->options = options;

#if MATTE_HAS_ONNXRUNTIME
    // TODO:
    // 1. 创建 Ort::Env / Ort::SessionOptions
    // 2. 按 useGpu 决定是否挂 CUDA execution provider
    // 3. 加载 BiRefNet ONNX 模型并缓存输入输出 tensor 名称
    impl_->initialized = true;
    if (errorMessage) {
        errorMessage->clear();
    }
    return true;
#else
    impl_->initialized = false;
    if (errorMessage) {
        *errorMessage = "ONNX Runtime is not linked yet. Set ONNXRUNTIME_ROOT to enable runtime loading.";
    }
    return false;
#endif
}

bool BiRefNetOnnxBackend::infer(const ImageTensor& input, MatteTensor& output, std::string* errorMessage) {
    if (!impl_->initialized) {
        if (errorMessage) {
            *errorMessage = "Backend is not initialized.";
        }
        return false;
    }

    if (input.size.width <= 0 || input.size.height <= 0 || input.data.empty()) {
        if (errorMessage) {
            *errorMessage = "Input image tensor is empty.";
        }
        return false;
    }

#if MATTE_HAS_ONNXRUNTIME
    // TODO:
    // 1. 将 Nuke 输入图像整理为 BiRefNet 需要的 NCHW float tensor
    // 2. 做 resize / normalize / channel reorder
    // 3. 执行 session.Run
    // 4. 将输出 matte resize 回原始分辨率
    output.width = input.size.width;
    output.height = input.size.height;
    output.alpha.resize(static_cast<size_t>(output.width) * static_cast<size_t>(output.height), 1.0f);
    if (errorMessage) {
        errorMessage->clear();
    }
    return true;
#else
    output.width = input.size.width;
    output.height = input.size.height;
    output.alpha.resize(static_cast<size_t>(output.width) * static_cast<size_t>(output.height));

    const int channels = std::max(1, input.size.channels);
    for (int y = 0; y < input.size.height; ++y) {
        for (int x = 0; x < input.size.width; ++x) {
            const size_t pixelIndex = static_cast<size_t>(y) * static_cast<size_t>(input.size.width) + static_cast<size_t>(x);
            const size_t base = pixelIndex * static_cast<size_t>(channels);
            const float r = input.data[base + 0];
            const float g = channels > 1 ? input.data[base + 1] : r;
            const float b = channels > 2 ? input.data[base + 2] : g;
            output.alpha[pixelIndex] = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f, 1.0f);
        }
    }

    if (errorMessage) {
        *errorMessage = "Using placeholder matte generation. Link ONNX Runtime and fill TODO sections for real BiRefNet inference.";
    }
    return true;
#endif
}

bool BiRefNetOnnxBackend::isInitialized() const {
    return impl_->initialized;
}

}  // namespace matte
