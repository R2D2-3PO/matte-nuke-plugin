#pragma once

#include "BiRefNetPlugin/MatteTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace matte {

class BiRefNetOnnxBackend {
public:
    BiRefNetOnnxBackend();
    ~BiRefNetOnnxBackend();

    bool initialize(const InferenceOptions& options, std::string* errorMessage);
    bool infer(const ImageTensor& input, MatteTensor& output, std::string* errorMessage);
    bool isInitialized() const;
    const InferenceOptions& options() const;
    const std::vector<std::string>& outputNames() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace matte
