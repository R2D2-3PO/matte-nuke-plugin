#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace matte {

struct ImageSize {
    int width = 0;
    int height = 0;
    int channels = 4;
};

struct ImageTensor {
    ImageSize size;
    std::vector<float> data;
};

struct MatteTensor {
    int width = 0;
    int height = 0;
    std::vector<float> alpha;
};

struct InferenceOptions {
    std::string modelPath;
    bool useGpu = false;
    int inputWidth = 1024;
    int inputHeight = 1024;
    float maskThreshold = 0.5f;
};

}  // namespace matte
