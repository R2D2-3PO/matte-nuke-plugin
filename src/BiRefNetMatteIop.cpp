#include "BiRefNetPlugin/BiRefNetMatteIop.h"

#include "DDImage/Knobs.h"
#include "DDImage/Row.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

using namespace DD::Image;

namespace {

constexpr const char* kClassName = "BiRefNetMatte";
constexpr const char* kHelpText =
    "BiRefNet matte extraction scaffold for Nuke 15.\n"
    "Runs a TorchScript BiRefNet model over the full image RoD or a selected box and writes the matte to alpha.";

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

bool sameOptions(const matte::InferenceOptions& lhs, const matte::InferenceOptions& rhs) {
    return lhs.modelPath == rhs.modelPath &&
           lhs.torchvisionOpsLibraryPath == rhs.torchvisionOpsLibraryPath &&
           lhs.useGpu == rhs.useGpu &&
           lhs.inputWidth == rhs.inputWidth &&
           lhs.inputHeight == rhs.inputHeight;
}

}  // namespace

const Iop::Description BiRefNetMatteIop::description(
    kClassName,
    "Merge/Matte/BiRefNetMatte",
    BiRefNetMatteIop::build);

BiRefNetMatteIop::BiRefNetMatteIop(Node* node)
    : Iop(node),
      modelPath_(),
#ifdef MATTE_DEFAULT_TORCHVISION_OPS_LIBRARY
      torchvisionOpsLibraryPath_(MATTE_DEFAULT_TORCHVISION_OPS_LIBRARY),
#else
      torchvisionOpsLibraryPath_(),
#endif
      useGpu_(false),
      passthrough_(false),
      useSelectionBox_(false),
      keepAlphaOutsideBox_(false),
      selectionBox_{0.0f, 0.0f, 1920.0f, 1080.0f},
      unpremultInput_(true),
      clampInput_(true),
      inputWidth_(2048),
      inputHeight_(2048),
      maskThreshold_(0.5f),
      backendAttempted_(false),
      backendReady_(false) {
    inputs(1);
}

BiRefNetMatteIop::~BiRefNetMatteIop() = default;

Iop* BiRefNetMatteIop::build(Node* node) {
    return new BiRefNetMatteIop(node);
}

const char* BiRefNetMatteIop::Class() const {
    return kClassName;
}

const char* BiRefNetMatteIop::node_help() const {
    return kHelpText;
}

void BiRefNetMatteIop::knobs(Knob_Callback callback) {
    Text_knob(callback, "BiRefNet uses LibTorch + TorchScript on the full image RoD or a selected box.");
    Divider(callback, "model");

    String_knob(callback, &modelPath_, "model_path", "model path");
    Tooltip(callback, "Path to the exported BiRefNet TorchScript model (.pt or .ts).");

    String_knob(callback, &torchvisionOpsLibraryPath_, "torchvision_ops_library", "torchvision ops");
    Tooltip(callback, "Optional path to torchvision/_C.so so deform_conv2d and other TorchVision custom ops are registered in Nuke. The build tries to auto-fill this on your machine.");

    Bool_knob(callback, &useGpu_, "use_gpu", "use GPU");
    Tooltip(callback, "Enable CUDA execution if your LibTorch build supports it.");

    Int_knob(callback, &inputWidth_, "input_width", "input width");
    SetRange(callback, 64, 4096);
    Tooltip(callback, "Model input width used before TorchScript inference.");

    Int_knob(callback, &inputHeight_, "input_height", "input height");
    SetRange(callback, 64, 4096);
    Tooltip(callback, "Model input height used before TorchScript inference.");

    Divider(callback, "preprocess");

    Bool_knob(callback, &useSelectionBox_, "use_selection_box", "use selection box");
    Tooltip(callback, "Run BiRefNet only on the selected image rectangle instead of the full image.");

    BBox_knob(callback, selectionBox_, "selection_box", "selection box");
    Tooltip(callback, "Viewer-draggable rectangle used as the BiRefNet crop when selection box mode is enabled.");

    Bool_knob(callback, &keepAlphaOutsideBox_, "keep_alpha_outside_box", "keep alpha outside box");
    Tooltip(callback, "When selection box mode is enabled, preserve the source alpha outside the box instead of writing zero.");

    Bool_knob(callback, &unpremultInput_, "unpremult_input", "unpremult input");
    Tooltip(callback, "Divide RGB by alpha before feeding the model.");

    Bool_knob(callback, &clampInput_, "clamp_input", "clamp input");
    Tooltip(callback, "Clamp RGB to 0..1 before normalization.");

    Divider(callback, "output");

    Bool_knob(callback, &passthrough_, "passthrough", "passthrough");
    Tooltip(callback, "Bypass the node and forward the source alpha.");

    Float_knob(callback, &maskThreshold_, "mask_threshold", "mask threshold");
    SetRange(callback, 0.0, 1.0);
    Tooltip(callback, "Applied only to the fallback placeholder matte when TorchScript inference is unavailable.");

}

void BiRefNetMatteIop::_validate(bool forReal) {
    copy_info();
    info_.turn_on(Mask_Alpha);
    set_out_channels(Mask_RGBA);
    if (forReal) {
        ensureBackendReady();
    }
    Iop::_validate(forReal);
}

void BiRefNetMatteIop::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
    if (input(0)) {
        ChannelSet requested = channels;
        requested += Mask_RGBA;
        input0().request(info_.x(), info_.y(), info_.r(), info_.t(), requested, count);
    }
}

void BiRefNetMatteIop::append(Hash& hash) {
    Iop::append(hash);
    hash.append(modelPath_.c_str());
    hash.append(torchvisionOpsLibraryPath_.c_str());
    hash.append(static_cast<int>(useGpu_));
    hash.append(static_cast<int>(passthrough_));
    hash.append(static_cast<int>(useSelectionBox_));
    hash.append(static_cast<int>(keepAlphaOutsideBox_));
    for (float value : selectionBox_) {
        hash.append(value);
    }
    hash.append(static_cast<int>(unpremultInput_));
    hash.append(static_cast<int>(clampInput_));
    hash.append(inputWidth_);
    hash.append(inputHeight_);
    hash.append(maskThreshold_);
}

void BiRefNetMatteIop::invalidateCache() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    cachedFrame_.alpha.clear();
    cachedFrame_.valid = false;
    cachedFrame_.modelPath.clear();
}

void BiRefNetMatteIop::setBackendStatus(const std::string& status, const std::string& error) {
    (void)status;
    (void)error;
}

void BiRefNetMatteIop::ensureBackendReady() {
    matte::InferenceOptions options;
    options.modelPath = modelPath_;
    options.torchvisionOpsLibraryPath = torchvisionOpsLibraryPath_;
    options.useGpu = useGpu_;
    options.inputWidth = inputWidth_;
    options.inputHeight = inputHeight_;

    if (options.modelPath.empty()) {
        backendAttempted_ = true;
        backendReady_ = false;
        backendError_ = "TorchScript model path is empty. Using placeholder matte.";
        setBackendStatus("fallback", backendError_);
        invalidateCache();
        return;
    }

    if (backendAttempted_ && sameOptions(backend_.options(), options)) {
        return;
    }

    backendAttempted_ = true;
    backendReady_ = backend_.initialize(options, &backendError_);
    setBackendStatus(backendReady_ ? "ready" : "fallback", backendReady_ ? std::string() : backendError_);
    invalidateCache();
}

bool BiRefNetMatteIop::buildInputTensor(const Box& box, matte::ImageTensor& inputTensor) const {
    const int width = box.w();
    const int height = box.h();
    if (width <= 0 || height <= 0 || !input(0)) {
        return false;
    }

    inputTensor.size.width = width;
    inputTensor.size.height = height;
    inputTensor.size.channels = 4;
    inputTensor.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

    Row srcRow(box.x(), box.r());
    ChannelSet sourceChannels = Mask_RGBA;

    for (int py = box.y(); py < box.t(); ++py) {
        input0().get(py, box.x(), box.r(), sourceChannels, srcRow);

        const float* srcR = srcRow[Chan_Red];
        const float* srcG = srcRow[Chan_Green];
        const float* srcB = srcRow[Chan_Blue];
        const float* srcA = srcRow[Chan_Alpha];
        const size_t rowBase = static_cast<size_t>(py - box.y()) * static_cast<size_t>(width) * 4;

        for (int px = box.x(); px < box.r(); ++px) {
            const size_t outBase = rowBase + static_cast<size_t>(px - box.x()) * 4;
            const float alpha = srcA ? srcA[px] : 1.0f;
            const float safeAlpha = alpha > 1e-6f ? alpha : 1.0f;
            float red = srcR ? srcR[px] : 0.0f;
            float green = srcG ? srcG[px] : 0.0f;
            float blue = srcB ? srcB[px] : 0.0f;

            if (unpremultInput_) {
                red /= safeAlpha;
                green /= safeAlpha;
                blue /= safeAlpha;
            }

            if (clampInput_) {
                red = clamp01(red);
                green = clamp01(green);
                blue = clamp01(blue);
            }

            inputTensor.data[outBase + 0] = red;
            inputTensor.data[outBase + 1] = green;
            inputTensor.data[outBase + 2] = blue;
            inputTensor.data[outBase + 3] = clamp01(alpha);
        }
    }

    return true;
}

bool BiRefNetMatteIop::getInferenceBox(Box& box) const {
    const Box sourceBox(info_.x(), info_.y(), info_.r(), info_.t());
    if (sourceBox.w() <= 0 || sourceBox.h() <= 0) {
        return false;
    }

    if (!useSelectionBox_) {
        box = sourceBox;
        return true;
    }

    const float left = std::min(selectionBox_[0], selectionBox_[2]);
    const float right = std::max(selectionBox_[0], selectionBox_[2]);
    const float bottom = std::min(selectionBox_[1], selectionBox_[3]);
    const float top = std::max(selectionBox_[1], selectionBox_[3]);

    int ix = static_cast<int>(std::floor(left));
    int iy = static_cast<int>(std::floor(bottom));
    int ir = static_cast<int>(std::ceil(right));
    int it = static_cast<int>(std::ceil(top));

    ix = std::max(ix, sourceBox.x());
    iy = std::max(iy, sourceBox.y());
    ir = std::min(ir, sourceBox.r());
    it = std::min(it, sourceBox.t());

    if (ir <= ix || it <= iy) {
        return false;
    }

    box.set(ix, iy, ir, it);
    return true;
}

bool BiRefNetMatteIop::ensureMaskCache(const Box& box) {
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        if (cachedFrame_.valid &&
            cachedFrame_.box.x() == box.x() &&
            cachedFrame_.box.y() == box.y() &&
            cachedFrame_.box.r() == box.r() &&
            cachedFrame_.box.t() == box.t() &&
            cachedFrame_.modelPath == modelPath_ &&
            cachedFrame_.torchvisionOpsLibraryPath == torchvisionOpsLibraryPath_ &&
            cachedFrame_.useGpu == useGpu_ &&
            cachedFrame_.inputWidth == inputWidth_ &&
            cachedFrame_.inputHeight == inputHeight_ &&
            cachedFrame_.unpremultInput == unpremultInput_ &&
            cachedFrame_.clampInput == clampInput_) {
            return true;
        }
    }

    std::lock_guard<std::mutex> inferenceLock(inferenceMutex_);

    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        if (cachedFrame_.valid &&
            cachedFrame_.box.x() == box.x() &&
            cachedFrame_.box.y() == box.y() &&
            cachedFrame_.box.r() == box.r() &&
            cachedFrame_.box.t() == box.t() &&
            cachedFrame_.modelPath == modelPath_ &&
            cachedFrame_.torchvisionOpsLibraryPath == torchvisionOpsLibraryPath_ &&
            cachedFrame_.useGpu == useGpu_ &&
            cachedFrame_.inputWidth == inputWidth_ &&
            cachedFrame_.inputHeight == inputHeight_ &&
            cachedFrame_.unpremultInput == unpremultInput_ &&
            cachedFrame_.clampInput == clampInput_) {
            return true;
        }
    }

    ensureBackendReady();
    if (!backendReady_) {
        setBackendStatus("fallback", backendError_);
        return false;
    }

    matte::ImageTensor inputTensor;
    if (!buildInputTensor(box, inputTensor)) {
        backendError_ = "Failed to gather input pixels from Nuke.";
        setBackendStatus("fallback", backendError_);
        return false;
    }

    matte::MatteTensor outputTensor;
    if (!backend_.infer(inputTensor, outputTensor, &backendError_)) {
        setBackendStatus("fallback", backendError_);
        return false;
    }

    if (outputTensor.width != box.w() || outputTensor.height != box.h() ||
        outputTensor.alpha.size() != static_cast<size_t>(box.w()) * static_cast<size_t>(box.h())) {
        backendError_ = "Backend returned an unexpected matte size.";
        setBackendStatus("fallback", backendError_);
        return false;
    }

    CachedMatteFrame frame;
    frame.box = box;
    frame.alpha = std::move(outputTensor.alpha);
    frame.modelPath = modelPath_;
    frame.torchvisionOpsLibraryPath = torchvisionOpsLibraryPath_;
    frame.useGpu = useGpu_;
    frame.inputWidth = inputWidth_;
    frame.inputHeight = inputHeight_;
    frame.unpremultInput = unpremultInput_;
    frame.clampInput = clampInput_;
    frame.maskThreshold = maskThreshold_;
    frame.valid = true;

    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (cachedFrame_.valid &&
        cachedFrame_.box.x() == box.x() &&
        cachedFrame_.box.y() == box.y() &&
        cachedFrame_.box.r() == box.r() &&
        cachedFrame_.box.t() == box.t() &&
        cachedFrame_.modelPath == modelPath_ &&
        cachedFrame_.torchvisionOpsLibraryPath == torchvisionOpsLibraryPath_ &&
        cachedFrame_.useGpu == useGpu_ &&
        cachedFrame_.inputWidth == inputWidth_ &&
        cachedFrame_.inputHeight == inputHeight_ &&
        cachedFrame_.unpremultInput == unpremultInput_ &&
        cachedFrame_.clampInput == clampInput_) {
        return true;
    }

    cachedFrame_ = std::move(frame);
    setBackendStatus("ready", std::string());
    return true;
}

void BiRefNetMatteIop::engine(int y, int x, int r, ChannelMask channels, Row& row) {
    if (!input(0)) {
        row.erase(channels);
        return;
    }

    Row sourceRow(x, r);
    input0().get(y, x, r, Mask_RGBA, sourceRow);

    ChannelSet passThroughChannels = channels;
    passThroughChannels -= Mask_Alpha;
    passThroughChannels &= Mask_RGB;
    if (passThroughChannels) {
        row.copy(sourceRow, passThroughChannels, x, r);
    }

    if (!(channels & Chan_Alpha)) {
        return;
    }

    const float* srcR = sourceRow[Chan_Red];
    const float* srcG = sourceRow[Chan_Green];
    const float* srcB = sourceRow[Chan_Blue];
    const float* srcA = sourceRow[Chan_Alpha];

    float* outAlpha = row.writable(Chan_Alpha);
    if (!outAlpha) {
        return;
    }

    if (passthrough_) {
        if (srcA) {
            std::memcpy(outAlpha + x, srcA + x, static_cast<size_t>(r - x) * sizeof(float));
        }
        return;
    }

    Box box;
    if (!getInferenceBox(box)) {
        for (int px = x; px < r; ++px) {
            outAlpha[px] = keepAlphaOutsideBox_ && srcA ? srcA[px] : 0.0f;
        }
        return;
    }

    const auto writeOutsideBoxAlpha = [&](int px) {
        outAlpha[px] = useSelectionBox_ && (!keepAlphaOutsideBox_ || !srcA) ? 0.0f : (srcA ? srcA[px] : 1.0f);
    };

    if (box.w() > 0 && box.h() > 0 && ensureMaskCache(box)) {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        const int cacheWidth = cachedFrame_.box.w();
        const int rowOffset = y - cachedFrame_.box.y();
        const bool rowInsideBox = rowOffset >= 0 && rowOffset < cachedFrame_.box.h();

        for (int px = x; px < r; ++px) {
            const int columnOffset = px - cachedFrame_.box.x();
            if (!rowInsideBox || columnOffset < 0 || columnOffset >= cacheWidth) {
                writeOutsideBoxAlpha(px);
                continue;
            }

            const size_t index = static_cast<size_t>(rowOffset) * static_cast<size_t>(cacheWidth) + static_cast<size_t>(columnOffset);
            outAlpha[px] = clamp01(cachedFrame_.alpha[index]);
        }
        return;
    }

    // Fallback while backend setup/inference fails.
    for (int px = x; px < r; ++px) {
        if (useSelectionBox_ && (px < box.x() || px >= box.r() || y < box.y() || y >= box.t())) {
            writeOutsideBoxAlpha(px);
            continue;
        }

        const float red = srcR ? srcR[px] : 0.0f;
        const float green = srcG ? srcG[px] : red;
        const float blue = srcB ? srcB[px] : green;
        const float luma = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
        const float matte = luma >= maskThreshold_ ? 1.0f : luma / std::max(maskThreshold_, 0.001f);
        outAlpha[px] = clamp01(matte);
    }
}
