// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include "newton_sdf_vtk_writer.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool cuda_check(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << "\n";
    return false;
}

std::string read_binary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot read test output: " + path.string());
    }
    return std::string(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>());
}

std::uint64_t read_u64(const std::string& bytes, std::size_t offset) {
    if (offset + sizeof(std::uint64_t) > bytes.size()) {
        throw std::runtime_error("Unexpected end of VTI appended data.");
    }
    std::uint64_t value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

float read_float(const std::string& bytes, std::size_t offset) {
    if (offset + sizeof(float) > bytes.size()) {
        throw std::runtime_error("Unexpected end of VTI scalar payload.");
    }
    float value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (nsv_has_cuda_support() != 1) {
        std::cerr << "CUDA VTI writer was not compiled with CUDA Runtime support.\n";
        return 1;
    }

    int device_count = 0;
    const cudaError_t device_result = cudaGetDeviceCount(&device_count);
    if (device_result != cudaSuccess || device_count == 0) {
        // CUDA compilation is still useful on CI workers without a physical GPU.
        cudaGetLastError();
        std::cout << "SKIP: CUDA Runtime is available but no CUDA device is present.\n";
        return 0;
    }

    const int device_ordinal = 0;
    if (!cuda_check(cudaSetDevice(device_ordinal), "cudaSetDevice")) {
        return 1;
    }

    const std::filesystem::path output_dir =
        argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path() / "newton-sdf-cuda-test";
    std::filesystem::create_directories(output_dir);
    const auto output_path = output_dir / "cuda-volume.vti";

    const std::vector<float> host_values{
        -1.0F, 1.0F,
        -0.5F, 0.5F,
        -0.25F, 0.25F,
        -0.1F, 0.1F,
    };
    float* device_values = nullptr;
    cudaStream_t stream = nullptr;
    if (!cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_values), host_values.size() * sizeof(float)), "cudaMalloc") ||
        !cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate") ||
        !cuda_check(
            cudaMemcpyAsync(
                device_values,
                host_values.data(),
                host_values.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync(H2D)")) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (device_values != nullptr) {
            cudaFree(device_values);
        }
        return 1;
    }

    const double origin[3]{-0.25, 0.5, 1.0};
    const double spacing[3]{0.125, 0.25, 0.5};
    char error[2048]{};
    const int result = nsv_write_vti_cuda(
        output_path.u8string().c_str(),
        2,
        2,
        2,
        origin,
        spacing,
        device_values,
        host_values.size(),
        device_ordinal,
        reinterpret_cast<void*>(stream),
        "signed_distance_m",
        error,
        sizeof(error));

    bool cleanup_ok = true;
    cleanup_ok &= cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(test)");
    cleanup_ok &= cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    cleanup_ok &= cuda_check(cudaFree(device_values), "cudaFree");
    if (result != 0) {
        std::cerr << "nsv_write_vti_cuda failed: " << error << "\n";
        return 1;
    }
    if (!cleanup_ok) {
        return 1;
    }

    const std::string bytes = read_binary(output_path);
    constexpr char marker[] = "<AppendedData encoding=\"raw\">_";
    const auto marker_position = bytes.find(marker);
    if (marker_position == std::string::npos) {
        std::cerr << "CUDA VTI output has no raw appended data section.\n";
        return 1;
    }
    const std::size_t payload = marker_position + sizeof(marker) - 1;
    if (read_u64(bytes, payload) != host_values.size() * sizeof(float)) {
        std::cerr << "CUDA VTI block header has an incorrect byte count.\n";
        return 1;
    }
    for (std::size_t i = 0; i < host_values.size(); ++i) {
        const float actual = read_float(bytes, payload + sizeof(std::uint64_t) + i * sizeof(float));
        if (actual != host_values[i]) {
            std::cerr << "CUDA VTI payload mismatch at index " << i << ".\n";
            return 1;
        }
    }

    std::cout << "CUDA VTI device-pointer test passed: " << output_path << "\n";
    return 0;
}
