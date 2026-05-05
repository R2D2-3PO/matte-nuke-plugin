#pragma once

#include "BiRefNetPlugin/BiRefNetTorchBackend.h"

#include "DDImage/Box.h"
#include "DDImage/Iop.h"

#include <mutex>
#include <string>
#include <vector>

namespace DD {
namespace Image {
class Knob;
class Row;
}  // namespace Image
}  // namespace DD
class Node;

class BiRefNetMatteIop final : public DD::Image::Iop {
public:
    explicit BiRefNetMatteIop(Node* node);
    ~BiRefNetMatteIop() override;

    void knobs(DD::Image::Knob_Callback callback) override;
    void _validate(bool forReal) override;
    void _request(int x, int y, int r, int t, DD::Image::ChannelMask channels, int count) override;
    void engine(int y, int x, int r, DD::Image::ChannelMask channels, DD::Image::Row& row) override;
    const char* Class() const override;
    const char* node_help() const override;

    static const DD::Image::Iop::Description description;
    static DD::Image::Iop* build(Node* node);

private:
    struct CachedMatteFrame {
        DD::Image::Box box;
        std::vector<float> alpha;
        std::string modelPath;
        std::string torchvisionOpsLibraryPath;
        bool useGpu = false;
        int inputWidth = 2048;
        int inputHeight = 2048;
        bool unpremultInput = true;
        bool clampInput = true;
        float maskThreshold = 0.5f;
        bool valid = false;
    };

    void append(DD::Image::Hash& hash) override;
    void ensureBackendReady();
    void setBackendStatus(const std::string& status, const std::string& error);
    bool ensureMaskCache(const DD::Image::Box& box);
    bool buildInputTensor(const DD::Image::Box& box, matte::ImageTensor& inputTensor) const;
    bool getInferenceBox(DD::Image::Box& box) const;
    void invalidateCache();

    std::string modelPath_;
    std::string torchvisionOpsLibraryPath_;
    bool useGpu_;
    bool passthrough_;
    bool useSelectionBox_;
    bool keepAlphaOutsideBox_;
    float selectionBox_[4];
    bool unpremultInput_;
    bool clampInput_;
    int inputWidth_;
    int inputHeight_;
    float maskThreshold_;

    mutable bool backendAttempted_;
    mutable bool backendReady_;
    mutable std::string backendError_;
    mutable matte::BiRefNetTorchBackend backend_;
    mutable std::mutex cacheMutex_;
    mutable std::mutex inferenceMutex_;
    mutable CachedMatteFrame cachedFrame_;
};
