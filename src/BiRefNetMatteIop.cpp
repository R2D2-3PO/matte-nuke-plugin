#include "BiRefNetPlugin/BiRefNetMatteIop.h"

#include "DDImage/Knobs.h"
#include "DDImage/Row.h"

#include <algorithm>
#include <cstring>

using namespace DD::Image;

namespace {

constexpr const char* kClassName = "BiRefNetMatte";
constexpr const char* kHelpText =
    "BiRefNet matte extraction scaffold for Nuke 15.\n"
    "Current version ships with a placeholder alpha generator and ONNX Runtime integration hooks.";

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

bool sameOptions(const matte::InferenceOptions& lhs, const matte::InferenceOptions& rhs) {
    return lhs.modelPath == rhs.modelPath &&
           lhs.useGpu == rhs.useGpu &&
           lhs.inputWidth == rhs.inputWidth &&
           lhs.inputHeight == rhs.inputHeight &&
           lhs.maskThreshold == rhs.maskThreshold;
}

}  // namespace

const Iop::Description BiRefNetMatteIop::description(
    kClassName,
    "Merge/Matte/BiRefNetMatte",
    BiRefNetMatteIop::build);

BiRefNetMatteIop::BiRefNetMatteIop(Node* node)
    : Iop(node),
      useGpu_(false),
      passthrough_(false),
      maskThreshold_(0.5f),
      backendAttempted_(false),
      backendReady_(false) {
    inputs(1);
    modelPath_[0] = '\0';
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
    File_knob(callback, modelPath_, "model_path", "model path");
    Tooltip(callback, "Path to the exported BiRefNet ONNX model.");

    Bool_knob(callback, &useGpu_, "use_gpu", "use GPU");
    Tooltip(callback, "Reserve this for CUDA / CoreML / TensorRT execution provider integration.");

    Bool_knob(callback, &passthrough_, "passthrough", "passthrough");
    Tooltip(callback, "Bypass the node and forward the source alpha.");

    Float_knob(callback, &maskThreshold_, "mask_threshold", "mask threshold");
    SetRange(callback, 0.0, 1.0);
    Tooltip(callback, "Applied to the placeholder matte path. Real BiRefNet logic can reuse or ignore it.");
}

void BiRefNetMatteIop::_validate(bool forReal) {
    copy_info();
    info_.turn_on(Mask_Alpha);
    set_out_channels(Mask_RGBA);
    Iop::_validate(forReal);
}

void BiRefNetMatteIop::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
    if (input0()) {
        input0().request(x, y, r, t, channels | Mask_RGBA, count);
    }
}

void BiRefNetMatteIop::append(Hash& hash) {
    Iop::append(hash);
    hash.append(modelPath_);
    hash.append(static_cast<int>(useGpu_));
    hash.append(static_cast<int>(passthrough_));
    hash.append(maskThreshold_);
}

void BiRefNetMatteIop::ensureBackendReady() {
    matte::InferenceOptions options;
    options.modelPath = modelPath_;
    options.useGpu = useGpu_;
    options.maskThreshold = maskThreshold_;

    if (options.modelPath.empty()) {
        backendAttempted_ = true;
        backendReady_ = false;
        backendError_ = "Model path is empty. Using placeholder matte.";
        return;
    }

    if (backendAttempted_ && sameOptions(backend_.options(), options)) {
        return;
    }

    backendAttempted_ = true;
    backendReady_ = backend_.initialize(options, &backendError_);
}

void BiRefNetMatteIop::engine(int y, int x, int r, ChannelMask channels, Row& row) {
    if (!input0()) {
        row.erase(channels);
        return;
    }

    input0().get(y, x, r, channels | Mask_RGBA, row);

    const float* srcR = row[Chan_Red];
    const float* srcG = row[Chan_Green];
    const float* srcB = row[Chan_Blue];
    const float* srcA = row[Chan_Alpha];

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

    ensureBackendReady();

    // 这里先保留一个稳定的占位路径，保证 Nuke 节点主干先可用。
    // 真正接 BiRefNet 时，建议把整帧采样、预处理、session.Run 和 mask cache 放到这里的上一层。
    for (int px = x; px < r; ++px) {
        const float red = srcR ? srcR[px] : 0.0f;
        const float green = srcG ? srcG[px] : red;
        const float blue = srcB ? srcB[px] : green;
        const float luma = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
        const float matte = luma >= maskThreshold_ ? 1.0f : luma / std::max(maskThreshold_, 0.001f);
        outAlpha[px] = clamp01(matte);
    }
}
