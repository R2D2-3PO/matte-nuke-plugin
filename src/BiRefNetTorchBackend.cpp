#include "BiRefNetPlugin/BiRefNetTorchBackend.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if MATTE_HAS_LIBTORCH
#include <dlfcn.h>
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace matte {

namespace {

constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[3] = {0.229f, 0.224f, 0.225f};

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float sampleChannelBilinear(const ImageTensor& image, int channel, float srcX, float srcY) {
    const int width = image.size.width;
    const int height = image.size.height;
    const int channels = std::max(1, image.size.channels);

    const float clampedX = std::max(0.0f, std::min(srcX, static_cast<float>(width - 1)));
    const float clampedY = std::max(0.0f, std::min(srcY, static_cast<float>(height - 1)));

    const int x0 = static_cast<int>(std::floor(clampedX));
    const int y0 = static_cast<int>(std::floor(clampedY));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);

    const float tx = clampedX - static_cast<float>(x0);
    const float ty = clampedY - static_cast<float>(y0);

    const int channelIndex = std::min(channel, channels - 1);
    const auto at = [&](int x, int y) {
        const size_t pixelIndex = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        return image.data[pixelIndex * static_cast<size_t>(channels) + static_cast<size_t>(channelIndex)];
    };

    const float v00 = at(x0, y0);
    const float v10 = at(x1, y0);
    const float v01 = at(x0, y1);
    const float v11 = at(x1, y1);

    const float top = v00 + (v10 - v00) * tx;
    const float bottom = v01 + (v11 - v01) * tx;
    return top + (bottom - top) * ty;
}

std::vector<float> preprocessToNchw(const ImageTensor& input, int dstWidth, int dstHeight) {
    std::vector<float> tensor(static_cast<size_t>(3) * static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight));

    const float scaleX = static_cast<float>(input.size.width) / static_cast<float>(dstWidth);
    const float scaleY = static_cast<float>(input.size.height) / static_cast<float>(dstHeight);

    for (int y = 0; y < dstHeight; ++y) {
        const float srcY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
        for (int x = 0; x < dstWidth; ++x) {
            const float srcX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
            const size_t hwIndex = static_cast<size_t>(y) * static_cast<size_t>(dstWidth) + static_cast<size_t>(x);

            for (int channel = 0; channel < 3; ++channel) {
                const float pixel = sampleChannelBilinear(input, channel, srcX, srcY);
                const float normalized = (pixel - kImageNetMean[channel]) / kImageNetStd[channel];
                tensor[static_cast<size_t>(channel) * static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight) + hwIndex] = normalized;
            }
        }
    }

    return tensor;
}

std::vector<float> resizeMaskToSource(const float* srcMask, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    std::vector<float> resized(static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight));

    const float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    const float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);

    for (int y = 0; y < dstHeight; ++y) {
        const float srcY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
        const float clampedY = std::max(0.0f, std::min(srcY, static_cast<float>(srcHeight - 1)));
        const int y0 = static_cast<int>(std::floor(clampedY));
        const int y1 = std::min(y0 + 1, srcHeight - 1);
        const float ty = clampedY - static_cast<float>(y0);

        for (int x = 0; x < dstWidth; ++x) {
            const float srcX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
            const float clampedX = std::max(0.0f, std::min(srcX, static_cast<float>(srcWidth - 1)));
            const int x0 = static_cast<int>(std::floor(clampedX));
            const int x1 = std::min(x0 + 1, srcWidth - 1);
            const float tx = clampedX - static_cast<float>(x0);

            const auto at = [&](int sx, int sy) {
                return srcMask[static_cast<size_t>(sy) * static_cast<size_t>(srcWidth) + static_cast<size_t>(sx)];
            };

            const float v00 = at(x0, y0);
            const float v10 = at(x1, y0);
            const float v01 = at(x0, y1);
            const float v11 = at(x1, y1);
            const float top = v00 + (v10 - v00) * tx;
            const float bottom = v01 + (v11 - v01) * tx;

            resized[static_cast<size_t>(y) * static_cast<size_t>(dstWidth) + static_cast<size_t>(x)] = clamp01(top + (bottom - top) * ty);
        }
    }

    return resized;
}

#if MATTE_HAS_LIBTORCH
torch::Tensor extractFinalTensor(const torch::jit::IValue& value) {
    if (value.isTensor()) {
        return value.toTensor();
    }
    if (value.isTensorList()) {
        const auto tensors = value.toTensorVector();
        return tensors.empty() ? torch::Tensor() : tensors.back();
    }
    if (value.isTuple()) {
        const auto& elements = value.toTuple()->elements();
        return elements.empty() ? torch::Tensor() : extractFinalTensor(elements.back());
    }
    if (value.isList()) {
        const auto list = value.toList();
        return list.empty() ? torch::Tensor() : extractFinalTensor(list.get(list.size() - 1));
    }
    return torch::Tensor();
}
#endif

}  // namespace

struct BiRefNetTorchBackend::Impl {
    bool initialized = false;
    InferenceOptions options;

#if MATTE_HAS_LIBTORCH
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;
    std::string loadedOpsLibrary;
    void* opsLibraryHandle = nullptr;
#endif
};

BiRefNetTorchBackend::BiRefNetTorchBackend() : impl_(std::make_unique<Impl>()) {}

BiRefNetTorchBackend::~BiRefNetTorchBackend() = default;

bool BiRefNetTorchBackend::initialize(const InferenceOptions& options, std::string* errorMessage) {
    impl_->options = options;

#if MATTE_HAS_LIBTORCH
    impl_->initialized = false;

    try {
        if (options.modelPath.empty()) {
            throw std::runtime_error("Model path is empty.");
        }
        if (!std::filesystem::exists(options.modelPath)) {
            throw std::runtime_error("TorchScript model file does not exist: " + options.modelPath);
        }
        if (options.inputWidth <= 0 || options.inputHeight <= 0) {
            throw std::runtime_error("Input resolution must be positive.");
        }

        if (!options.torchvisionOpsLibraryPath.empty() &&
            impl_->loadedOpsLibrary != options.torchvisionOpsLibraryPath) {
            if (!std::filesystem::exists(options.torchvisionOpsLibraryPath)) {
                throw std::runtime_error("TorchVision ops library does not exist: " + options.torchvisionOpsLibraryPath);
            }
            void* handle = dlopen(options.torchvisionOpsLibraryPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                throw std::runtime_error(std::string("Failed to load TorchVision ops library: ") + dlerror());
            }
            impl_->opsLibraryHandle = handle;
            impl_->loadedOpsLibrary = options.torchvisionOpsLibraryPath;
        }

        impl_->device = torch::kCPU;
        if (options.useGpu && torch::cuda::is_available()) {
            impl_->device = torch::Device(torch::kCUDA, 0);
        }

        impl_->module = torch::jit::load(options.modelPath, impl_->device);
        impl_->module.eval();
        impl_->initialized = true;

        if (errorMessage) {
            errorMessage->clear();
        }
        return true;
    } catch (const c10::Error& ex) {
        if (errorMessage) {
            *errorMessage = std::string("LibTorch initialize failed: ") + ex.what_without_backtrace();
        }
        return false;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = ex.what();
        }
        return false;
    }
#else
    impl_->initialized = false;
    if (errorMessage) {
        *errorMessage = "LibTorch is not linked yet. Set TORCH_CMAKE_PREFIX to enable TorchScript loading.";
    }
    return false;
#endif
}

bool BiRefNetTorchBackend::infer(const ImageTensor& input, MatteTensor& output, std::string* errorMessage) {
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

#if MATTE_HAS_LIBTORCH
    try {
        torch::InferenceMode guard;

        std::vector<float> inputTensorData = preprocessToNchw(input, impl_->options.inputWidth, impl_->options.inputHeight);
        const std::vector<int64_t> inputShape = {
            1,
            3,
            static_cast<int64_t>(impl_->options.inputHeight),
            static_cast<int64_t>(impl_->options.inputWidth),
        };

        torch::Tensor inputTensor = torch::from_blob(
            inputTensorData.data(),
            inputShape,
            torch::TensorOptions().dtype(torch::kFloat32)).clone().to(impl_->device);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(inputTensor);

        torch::jit::IValue outputValue = impl_->module.forward(inputs);
        torch::Tensor outputTensor = extractFinalTensor(outputValue);
        if (!outputTensor.defined()) {
            throw std::runtime_error("TorchScript model did not return a usable tensor.");
        }

        torch::Tensor logits = outputTensor;
        if (logits.dim() == 4) {
            logits = logits.squeeze(0);
        }
        if (logits.dim() == 3) {
            logits = logits.squeeze(0);
        }
        if (logits.dim() != 2) {
            throw std::runtime_error("Unexpected output tensor shape from TorchScript model.");
        }

        torch::Tensor maskTensor = torch::sigmoid(logits).to(torch::kCPU, torch::kFloat32).contiguous();
        const int outputHeight = static_cast<int>(maskTensor.size(0));
        const int outputWidth = static_cast<int>(maskTensor.size(1));
        const float* outputData = maskTensor.data_ptr<float>();

        output.width = input.size.width;
        output.height = input.size.height;
        output.alpha = resizeMaskToSource(outputData, outputWidth, outputHeight, output.width, output.height);

        if (errorMessage) {
            errorMessage->clear();
        }
        return true;
    } catch (const c10::Error& ex) {
        if (errorMessage) {
            *errorMessage = std::string("LibTorch inference failed: ") + ex.what_without_backtrace();
        }
        return false;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = ex.what();
        }
        return false;
    }
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
        *errorMessage = "Using placeholder matte generation. Link LibTorch and export a TorchScript model for real BiRefNet inference.";
    }
    return true;
#endif
}

bool BiRefNetTorchBackend::isInitialized() const {
    return impl_->initialized;
}

const InferenceOptions& BiRefNetTorchBackend::options() const {
    return impl_->options;
}

}  // namespace matte
