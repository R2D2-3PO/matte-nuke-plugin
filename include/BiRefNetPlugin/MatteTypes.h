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
    std::string torchvisionOpsLibraryPath;
    bool useGpu = false;
    int inputWidth = 2048;
    int inputHeight = 2048;
    float maskThreshold = 0.5f;
};

}  // namespace matte
