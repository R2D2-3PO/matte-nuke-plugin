#include "BiRefNetPlugin/BiRefNetOnnxBackend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#if MATTE_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#if MATTE_BUILD_WITH_CUDA
#include <cuda_provider_factory.h>
#endif
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

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float expNeg = std::exp(-value);
        return 1.0f / (1.0f + expNeg);
    }
    const float expPos = std::exp(value);
    return expPos / (1.0f + expPos);
}

}  // namespace

struct BiRefNetOnnxBackend::Impl {
    bool initialized = false;
    InferenceOptions options;

#if MATTE_HAS_ONNXRUNTIME
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "BiRefNetMatte"};
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::vector<std::string> outputNames;
#endif
};

BiRefNetOnnxBackend::BiRefNetOnnxBackend() : impl_(std::make_unique<Impl>()) {}

BiRefNetOnnxBackend::~BiRefNetOnnxBackend() = default;

bool BiRefNetOnnxBackend::initialize(const InferenceOptions& options, std::string* errorMessage) {
    impl_->options = options;

#if MATTE_HAS_ONNXRUNTIME
    impl_->initialized = false;
    impl_->session.reset();
    impl_->inputName.clear();
    impl_->outputNames.clear();

    try {
        if (options.modelPath.empty()) {
            throw std::runtime_error("Model path is empty.");
        }

        if (!std::filesystem::exists(options.modelPath)) {
            throw std::runtime_error("Model file does not exist: " + options.modelPath);
        }

        if (options.inputWidth <= 0 || options.inputHeight <= 0) {
            throw std::runtime_error("Input resolution must be positive.");
        }

        impl_->sessionOptions = std::make_unique<Ort::SessionOptions>();
        impl_->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->sessionOptions->SetIntraOpNumThreads(1);
        impl_->sessionOptions->SetInterOpNumThreads(1);

#if MATTE_BUILD_WITH_CUDA
        if (options.useGpu) {
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            impl_->sessionOptions->AppendExecutionProvider_CUDA(cudaOptions);
        }
#else
        (void)options.useGpu;
#endif

        impl_->session = std::make_unique<Ort::Session>(impl_->env, options.modelPath.c_str(), *impl_->sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = impl_->session->GetInputNameAllocated(0, allocator);
        impl_->inputName = inputName.get();

        const size_t outputCount = impl_->session->GetOutputCount();
        if (outputCount == 0) {
            throw std::runtime_error("Model has no outputs.");
        }

        impl_->outputNames.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto outputName = impl_->session->GetOutputNameAllocated(i, allocator);
            impl_->outputNames.emplace_back(outputName.get());
        }

        impl_->initialized = true;

        if (errorMessage) {
            errorMessage->clear();
        }
        return true;
    } catch (const Ort::Exception& ex) {
        if (errorMessage) {
            *errorMessage = std::string("ONNX Runtime initialize failed: ") + ex.what();
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
    try {
        std::vector<float> inputTensorData = preprocessToNchw(input, impl_->options.inputWidth, impl_->options.inputHeight);

        const std::array<int64_t, 4> inputShape = {
            1,
            3,
            static_cast<int64_t>(impl_->options.inputHeight),
            static_cast<int64_t>(impl_->options.inputWidth),
        };

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorData.data(),
            inputTensorData.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputNames[] = {impl_->inputName.c_str()};
        std::vector<const char*> outputNames;
        outputNames.reserve(impl_->outputNames.size());
        for (const std::string& name : impl_->outputNames) {
            outputNames.push_back(name.c_str());
        }

        auto outputValues = impl_->session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames.data(),
            outputNames.size());

        if (outputValues.empty()) {
            throw std::runtime_error("Model returned no outputs.");
        }

        const Ort::Value& outputTensor = outputValues.back();
        if (!outputTensor.IsTensor()) {
            throw std::runtime_error("Final model output is not a tensor.");
        }

        const auto tensorInfo = outputTensor.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> outputShape = tensorInfo.GetShape();
        const float* outputData = outputTensor.GetTensorData<float>();

        if (outputShape.size() < 4) {
            throw std::runtime_error("Unexpected output rank. Expected [N,C,H,W]-like tensor.");
        }

        const int outputHeight = static_cast<int>(outputShape[outputShape.size() - 2]);
        const int outputWidth = static_cast<int>(outputShape[outputShape.size() - 1]);
        if (outputWidth <= 0 || outputHeight <= 0) {
            throw std::runtime_error("Unexpected output tensor shape.");
        }

        std::vector<float> mask(static_cast<size_t>(outputWidth) * static_cast<size_t>(outputHeight));
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i] = sigmoid(outputData[i]);
        }

        output.width = input.size.width;
        output.height = input.size.height;
        output.alpha = resizeMaskToSource(mask.data(), outputWidth, outputHeight, output.width, output.height);

        if (errorMessage) {
            errorMessage->clear();
        }
        return true;
    } catch (const Ort::Exception& ex) {
        if (errorMessage) {
            *errorMessage = std::string("ONNX Runtime inference failed: ") + ex.what();
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
        *errorMessage = "Using placeholder matte generation. Link ONNX Runtime and fill TODO sections for real BiRefNet inference.";
    }
    return true;
#endif
}

bool BiRefNetOnnxBackend::isInitialized() const {
    return impl_->initialized;
}

const InferenceOptions& BiRefNetOnnxBackend::options() const {
    return impl_->options;
}

const std::vector<std::string>& BiRefNetOnnxBackend::outputNames() const {
#if MATTE_HAS_ONNXRUNTIME
    return impl_->outputNames;
#else
    static const std::vector<std::string> kEmpty;
    return kEmpty;
#endif
}

}  // namespace matte
