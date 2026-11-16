// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include "newton_sdf_vtk_writer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(NSV_WITH_CUDA)
#  include <cuda_runtime_api.h>
#endif

namespace {

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

void set_error(char* buffer, std::size_t capacity, const std::string& message) noexcept {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    const std::size_t count = message.size() < capacity - 1 ? message.size() : capacity - 1;
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

std::filesystem::path make_path(const char* path_utf8) {
    if (path_utf8 == nullptr || path_utf8[0] == '\0') {
        throw std::invalid_argument("Output path must not be empty.");
    }
    return std::filesystem::u8path(path_utf8);
}

void ensure_parent_directory(const std::filesystem::path& path) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string escape_xml(const char* value) {
    if (value == nullptr) {
        throw std::invalid_argument("XML string must not be null.");
    }
    std::string result;
    for (const char ch : std::string(value)) {
        switch (ch) {
            case '&': result += "&amp;"; break;
            case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break;
            case '"': result += "&quot;"; break;
            case '\'': result += "&apos;"; break;
            default: result += ch; break;
        }
    }
    return result;
}

std::uint64_t checked_multiply(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error(std::string("Size overflow while calculating ") + label + ".");
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        throw std::overflow_error(std::string("Offset overflow while calculating ") + label + ".");
    }
    return a + b;
}

void write_bytes(std::ofstream& stream, const void* data, std::uint64_t byte_count) {
    if (byte_count == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("A non-empty VTK array has a null data pointer.");
    }
    constexpr std::uint64_t kMaxStreamWrite =
        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max());
    const auto* bytes = static_cast<const char*>(data);
    std::uint64_t remaining = byte_count;
    while (remaining > 0) {
        const auto chunk = remaining > kMaxStreamWrite ? kMaxStreamWrite : remaining;
        stream.write(bytes, static_cast<std::streamsize>(chunk));
        if (!stream) {
            throw std::runtime_error("Failed while writing VTK binary payload.");
        }
        bytes += chunk;
        remaining -= chunk;
    }
}

template <typename T>
void write_little_endian(std::ofstream& stream, const T* values, std::uint64_t count) {
    static_assert(std::is_trivially_copyable<T>::value, "VTK payload types must be trivially copyable.");
    const std::uint64_t byte_count = checked_multiply(count, sizeof(T), "array byte count");
    if (byte_count == 0) {
        return;
    }
    if (values == nullptr) {
        throw std::invalid_argument("A non-empty VTK array has a null data pointer.");
    }
    if (host_is_little_endian() || sizeof(T) == 1) {
        write_bytes(stream, values, byte_count);
        return;
    }

    std::array<char, sizeof(T)> item{};
    const auto* source = reinterpret_cast<const unsigned char*>(values);
    for (std::uint64_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < sizeof(T); ++j) {
            item[j] = static_cast<char>(source[i * sizeof(T) + (sizeof(T) - 1 - j)]);
        }
        stream.write(item.data(), static_cast<std::streamsize>(item.size()));
    }
    if (!stream) {
        throw std::runtime_error("Failed while byte-swapping a VTK binary payload.");
    }
}

void write_block_header(std::ofstream& stream, std::uint64_t byte_count) {
    write_little_endian(stream, &byte_count, 1);
}

void write_atomic(
    const std::filesystem::path& target,
    const std::function<void(std::ofstream&)>& writer) {
    ensure_parent_directory(target);
    auto temporary = target;
    temporary += ".tmp";

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);

    try {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("Cannot open temporary output file: " + temporary.string());
        }
        writer(stream);
        stream.flush();
        if (!stream) {
            throw std::runtime_error("Failed to flush temporary output file: " + temporary.string());
        }
        stream.close();
        if (!stream) {
            throw std::runtime_error("Failed to close temporary output file: " + temporary.string());
        }

        std::error_code rename_error;
        std::filesystem::rename(temporary, target, rename_error);
        if (rename_error) {
            // Windows does not replace an existing destination with rename().
            std::error_code remove_error;
            std::filesystem::remove(target, remove_error);
            rename_error.clear();
            std::filesystem::rename(temporary, target, rename_error);
        }
        if (rename_error) {
            throw std::runtime_error("Cannot publish output file: " + rename_error.message());
        }
    } catch (...) {
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void write_xml_prefix(std::ofstream& stream, const char* type) {
    stream << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"" << type
           << "\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
}

void write_appended_suffix(std::ofstream& stream) {
    stream << "\n</AppendedData>\n</VTKFile>\n";
}

void write_vtp_impl(
    const std::filesystem::path& path,
    const float* points,
    std::uint64_t point_count,
    const std::int64_t* triangles,
    std::uint64_t triangle_count) {
    const std::uint64_t point_values = checked_multiply(point_count, 3, "point component count");
    const std::uint64_t connectivity_values = checked_multiply(triangle_count, 3, "triangle connectivity count");
    const std::uint64_t point_bytes = checked_multiply(point_values, sizeof(float), "point bytes");
    const std::uint64_t connectivity_bytes =
        checked_multiply(connectivity_values, sizeof(std::int64_t), "connectivity bytes");
    const std::uint64_t offsets_bytes = checked_multiply(triangle_count, sizeof(std::int64_t), "offset bytes");
    if (connectivity_values > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("VTP polygon offsets exceed the Int64 VTK limit.");
    }

    const std::uint64_t points_offset = 0;
    const std::uint64_t connectivity_offset = checked_add(sizeof(std::uint64_t), point_bytes, "connectivity offset");
    const std::uint64_t offsets_offset = checked_add(
        checked_add(connectivity_offset, sizeof(std::uint64_t), "offsets offset"),
        connectivity_bytes,
        "offsets offset");

    write_atomic(path, [&](std::ofstream& stream) {
        write_xml_prefix(stream, "PolyData");
        stream << "  <PolyData>\n"
               << "    <Piece NumberOfPoints=\"" << point_count
               << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
               << triangle_count << "\">\n"
               << "      <PointData/>\n"
               << "      <CellData/>\n"
               << "      <Points>\n"
               << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
               << points_offset << "\"/>\n"
               << "      </Points>\n"
               << "      <Verts/>\n"
               << "      <Lines/>\n"
               << "      <Strips/>\n"
               << "      <Polys>\n"
               << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" offset=\""
               << connectivity_offset << "\"/>\n"
               << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" offset=\""
               << offsets_offset << "\"/>\n"
               << "      </Polys>\n"
               << "    </Piece>\n"
               << "  </PolyData>\n"
               << "  <AppendedData encoding=\"raw\">_";

        write_block_header(stream, point_bytes);
        write_little_endian(stream, points, point_values);

        write_block_header(stream, connectivity_bytes);
        write_little_endian(stream, triangles, connectivity_values);

        write_block_header(stream, offsets_bytes);
        for (std::uint64_t i = 0; i < triangle_count; ++i) {
            const std::int64_t offset = static_cast<std::int64_t>((i + 1) * 3);
            write_little_endian(stream, &offset, 1);
        }

        write_appended_suffix(stream);
    });
}

void write_vti_impl(
    const std::filesystem::path& path,
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const double* origin,
    const double* spacing,
    const float* values,
    std::uint64_t value_count,
    const char* array_name) {
    const std::uint64_t expected = checked_multiply(
        checked_multiply(nx, ny, "VTI dimensions"), nz, "VTI dimensions");
    if (expected != value_count) {
        std::ostringstream message;
        message << "VTI value count mismatch: expected " << expected << ", received " << value_count << ".";
        throw std::invalid_argument(message.str());
    }
    if (nx == 0 || ny == 0 || nz == 0) {
        throw std::invalid_argument("VTI dimensions must be non-zero.");
    }
    if (origin == nullptr || spacing == nullptr) {
        throw std::invalid_argument("VTI origin and spacing pointers must not be null.");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(origin[axis])) {
            throw std::invalid_argument("VTI origin must contain only finite values.");
        }
        if (!std::isfinite(spacing[axis]) || !(spacing[axis] > 0.0)) {
            throw std::invalid_argument("VTI spacing must be finite and positive on every axis.");
        }
    }

    const std::uint64_t value_bytes = checked_multiply(value_count, sizeof(float), "VTI value bytes");
    const std::string escaped_name = escape_xml(array_name == nullptr ? "scalars" : array_name);
    std::ostringstream extent;
    extent << "0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1);

    write_atomic(path, [&](std::ofstream& stream) {
        stream << std::setprecision(17);
        write_xml_prefix(stream, "ImageData");
        stream << "  <ImageData WholeExtent=\"" << extent.str() << "\" Origin=\""
               << origin[0] << " " << origin[1] << " " << origin[2] << "\" Spacing=\""
               << spacing[0] << " " << spacing[1] << " " << spacing[2] << "\">\n"
               << "    <Piece Extent=\"" << extent.str() << "\">\n"
               << "      <PointData Scalars=\"" << escaped_name << "\">\n"
               << "        <DataArray type=\"Float32\" Name=\"" << escaped_name
               << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n"
               << "      </PointData>\n"
               << "      <CellData/>\n"
               << "    </Piece>\n"
               << "  </ImageData>\n"
               << "  <AppendedData encoding=\"raw\">_";
        write_block_header(stream, value_bytes);
        write_little_endian(stream, values, value_count);
        write_appended_suffix(stream);
    });
}

#if defined(NSV_WITH_CUDA)

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device_ordinal) {
        check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
        check_cuda(cudaSetDevice(device_ordinal), "cudaSetDevice");
        active_ = true;
    }

    ~CudaDeviceGuard() {
        if (active_) {
            cudaSetDevice(previous_device_);
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
    int previous_device_ = 0;
    bool active_ = false;
};

class PinnedBuffers {
public:
    PinnedBuffers(std::size_t capacity, cudaStream_t stream) : stream_(stream) {
        check_cuda(cudaHostAlloc(&buffers_[0], capacity, cudaHostAllocPortable), "cudaHostAlloc(buffer 0)");
        try {
            check_cuda(cudaHostAlloc(&buffers_[1], capacity, cudaHostAllocPortable), "cudaHostAlloc(buffer 1)");
        } catch (...) {
            cudaFreeHost(buffers_[0]);
            buffers_[0] = nullptr;
            throw;
        }
    }

    ~PinnedBuffers() {
        // An exception may leave the second copy in flight; never free its
        // destination until the Warp stream has finished using it.
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
        for (void*& buffer : buffers_) {
            if (buffer != nullptr) {
                cudaFreeHost(buffer);
                buffer = nullptr;
            }
        }
    }

    void* operator[](std::size_t index) const { return buffers_[index]; }

    PinnedBuffers(const PinnedBuffers&) = delete;
    PinnedBuffers& operator=(const PinnedBuffers&) = delete;

private:
    void* buffers_[2]{nullptr, nullptr};
    cudaStream_t stream_ = nullptr;
};

void write_vti_cuda_impl(
    const std::filesystem::path& path,
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const double* origin,
    const double* spacing,
    const void* device_values,
    std::uint64_t value_count,
    int device_ordinal,
    void* stream_handle,
    const char* array_name) {
    const std::uint64_t expected = checked_multiply(
        checked_multiply(nx, ny, "VTI dimensions"), nz, "VTI dimensions");
    if (expected != value_count) {
        std::ostringstream message;
        message << "VTI value count mismatch: expected " << expected << ", received " << value_count << ".";
        throw std::invalid_argument(message.str());
    }
    if (nx == 0 || ny == 0 || nz == 0) {
        throw std::invalid_argument("VTI dimensions must be non-zero.");
    }
    if (origin == nullptr || spacing == nullptr) {
        throw std::invalid_argument("VTI origin and spacing pointers must not be null.");
    }
    if (device_values == nullptr) {
        throw std::invalid_argument("CUDA VTI device pointer must not be null.");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(origin[axis])) {
            throw std::invalid_argument("VTI origin must contain only finite values.");
        }
        if (!std::isfinite(spacing[axis]) || !(spacing[axis] > 0.0)) {
            throw std::invalid_argument("VTI spacing must be finite and positive on every axis.");
        }
    }

    const std::uint64_t value_bytes = checked_multiply(value_count, sizeof(float), "VTI value bytes");
    const std::string escaped_name = escape_xml(array_name == nullptr ? "scalars" : array_name);
    std::ostringstream extent;
    extent << "0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1);

    CudaDeviceGuard device_guard(device_ordinal);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_handle);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    check_cuda(cudaStreamIsCapturing(cuda_stream, &capture_status), "cudaStreamIsCapturing");
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::runtime_error("Native VTI file export cannot run while the Warp CUDA stream is being captured.");
    }

    constexpr std::uint64_t kChunkBytes = 64ULL * 1024ULL * 1024ULL;
    const auto buffer_capacity = static_cast<std::size_t>(
        std::min<std::uint64_t>(value_bytes, kChunkBytes));
    PinnedBuffers pinned_buffers(buffer_capacity, cuda_stream);
    const auto* source = static_cast<const char*>(device_values);

    write_atomic(path, [&](std::ofstream& file) {
        file << std::setprecision(17);
        write_xml_prefix(file, "ImageData");
        file << "  <ImageData WholeExtent=\"" << extent.str() << "\" Origin=\""
             << origin[0] << " " << origin[1] << " " << origin[2] << "\" Spacing=\""
             << spacing[0] << " " << spacing[1] << " " << spacing[2] << "\">\n"
             << "    <Piece Extent=\"" << extent.str() << "\">\n"
             << "      <PointData Scalars=\"" << escaped_name << "\">\n"
             << "        <DataArray type=\"Float32\" Name=\"" << escaped_name
             << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n"
             << "      </PointData>\n"
             << "      <CellData/>\n"
             << "    </Piece>\n"
             << "  </ImageData>\n"
             << "  <AppendedData encoding=\"raw\">_";
        write_block_header(file, value_bytes);

        std::uint64_t offset = 0;
        std::uint64_t current_bytes = std::min<std::uint64_t>(value_bytes, buffer_capacity);
        std::size_t current_buffer = 0;
        check_cuda(
            cudaMemcpyAsync(
                pinned_buffers[current_buffer],
                source,
                static_cast<std::size_t>(current_bytes),
                cudaMemcpyDeviceToHost,
                cuda_stream),
            "cudaMemcpyAsync(first VTI chunk)");
        check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize(first VTI chunk)");

        while (offset < value_bytes) {
            const std::uint64_t next_offset = offset + current_bytes;
            const bool has_next = next_offset < value_bytes;
            const std::size_t next_buffer = 1 - current_buffer;
            std::uint64_t next_bytes = 0;

            if (has_next) {
                next_bytes = std::min<std::uint64_t>(value_bytes - next_offset, buffer_capacity);
                check_cuda(
                    cudaMemcpyAsync(
                        pinned_buffers[next_buffer],
                        source + next_offset,
                        static_cast<std::size_t>(next_bytes),
                        cudaMemcpyDeviceToHost,
                        cuda_stream),
                    "cudaMemcpyAsync(next VTI chunk)");
            }

            // The next D2H transfer overlaps with writing the previous pinned
            // buffer to disk. Only two bounded pinned buffers are retained.
            write_bytes(file, pinned_buffers[current_buffer], current_bytes);
            offset = next_offset;

            if (has_next) {
                check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize(next VTI chunk)");
                current_buffer = next_buffer;
                current_bytes = next_bytes;
            }
        }

        write_appended_suffix(file);
    });
}

#endif  // NSV_WITH_CUDA

void write_vtm_impl(
    const std::filesystem::path& path,
    const char* const* names,
    const char* const* files,
    std::uint64_t count) {
    if (count > 0 && (names == nullptr || files == nullptr)) {
        throw std::invalid_argument("VTM dataset name and file arrays must not be null.");
    }

    write_atomic(path, [&](std::ofstream& stream) {
        stream << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
               << "  <vtkMultiBlockDataSet>\n";
        for (std::uint64_t i = 0; i < count; ++i) {
            const std::string name = escape_xml(names[i]);
            const std::string file = escape_xml(files[i]);
            stream << "    <DataSet index=\"" << i << "\" name=\"" << name
                   << "\" file=\"" << file << "\"/>\n";
        }
        stream << "  </vtkMultiBlockDataSet>\n"
               << "</VTKFile>\n";
    });
}

template <typename Function>
int invoke(char* error_buffer, std::size_t error_capacity, Function&& function) noexcept {
    try {
        function();
        set_error(error_buffer, error_capacity, "");
        return 0;
    } catch (const std::exception& error) {
        set_error(error_buffer, error_capacity, error.what());
    } catch (...) {
        set_error(error_buffer, error_capacity, "Unknown native VTK writer error.");
    }
    return -1;
}

}  // namespace

int nsv_has_cuda_support(void) {
#if defined(NSV_WITH_CUDA)
    return 1;
#else
    return 0;
#endif
}

int nsv_write_vti_cuda(
    const char* path_utf8,
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const double origin_xyz[3],
    const double spacing_xyz[3],
    const void* device_values,
    std::uint64_t value_count,
    std::int32_t device_ordinal,
    void* cuda_stream,
    const char* array_name_utf8,
    char* error_buffer,
    std::size_t error_capacity) {
#if defined(NSV_WITH_CUDA)
    return invoke(error_buffer, error_capacity, [&]() {
        write_vti_cuda_impl(
            make_path(path_utf8),
            nx,
            ny,
            nz,
            origin_xyz,
            spacing_xyz,
            device_values,
            value_count,
            device_ordinal,
            cuda_stream,
            array_name_utf8);
    });
#else
    (void)path_utf8;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)origin_xyz;
    (void)spacing_xyz;
    (void)device_values;
    (void)value_count;
    (void)device_ordinal;
    (void)cuda_stream;
    (void)array_name_utf8;
    return invoke(error_buffer, error_capacity, []() {
        throw std::runtime_error("This native VTK writer was built without CUDA Runtime support.");
    });
#endif
}

int nsv_write_vtp(
    const char* path_utf8,
    const float* points_xyz,
    std::uint64_t point_count,
    const std::int64_t* triangles,
    std::uint64_t triangle_count,
    char* error_buffer,
    std::size_t error_capacity) {
    return invoke(error_buffer, error_capacity, [&]() {
        if (point_count > 0 && points_xyz == nullptr) {
            throw std::invalid_argument("VTP point pointer must not be null.");
        }
        if (triangle_count > 0 && triangles == nullptr) {
            throw std::invalid_argument("VTP triangle pointer must not be null.");
        }
        write_vtp_impl(make_path(path_utf8), points_xyz, point_count, triangles, triangle_count);
    });
}

int nsv_write_vti(
    const char* path_utf8,
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const double origin_xyz[3],
    const double spacing_xyz[3],
    const float* values,
    std::uint64_t value_count,
    const char* array_name_utf8,
    char* error_buffer,
    std::size_t error_capacity) {
    return invoke(error_buffer, error_capacity, [&]() {
        if (value_count > 0 && values == nullptr) {
            throw std::invalid_argument("VTI scalar pointer must not be null.");
        }
        write_vti_impl(
            make_path(path_utf8), nx, ny, nz, origin_xyz, spacing_xyz, values, value_count, array_name_utf8);
    });
}

int nsv_write_vtm(
    const char* path_utf8,
    const char* const* dataset_names_utf8,
    const char* const* relative_files_utf8,
    std::uint64_t dataset_count,
    char* error_buffer,
    std::size_t error_capacity) {
    return invoke(error_buffer, error_capacity, [&]() {
        write_vtm_impl(make_path(path_utf8), dataset_names_utf8, relative_files_utf8, dataset_count);
    });
}
