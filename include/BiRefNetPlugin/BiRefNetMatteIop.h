#pragma once

#include "BiRefNetPlugin/BiRefNetOnnxBackend.h"

#include "DDImage/Iop.h"

#include <string>

namespace DD {
namespace Image {
class Knob;
class Row;
}  // namespace Image
}  // namespace DD

class BiRefNetMatteIop final : public DD::Image::Iop {
public:
    explicit BiRefNetMatteIop(DD::Image::Node* node);
    ~BiRefNetMatteIop() override;

    void knobs(DD::Image::Knob_Callback callback) override;
    void _validate(bool forReal) override;
    void _request(int x, int y, int r, int t, DD::Image::ChannelMask channels, int count) override;
    void engine(int y, int x, int r, DD::Image::ChannelMask channels, DD::Image::Row& row) override;
    const char* Class() const override;
    const char* node_help() const override;

    static const DD::Image::Iop::Description description;
    static DD::Image::Iop* build(DD::Image::Node* node);

private:
    void append(DD::Image::Hash& hash) override;
    void ensureBackendReady();

    char modelPath_[1024];
    bool useGpu_;
    bool passthrough_;
    float maskThreshold_;

    mutable bool backendAttempted_;
    mutable bool backendReady_;
    mutable std::string backendError_;
    mutable matte::BiRefNetOnnxBackend backend_;
};
