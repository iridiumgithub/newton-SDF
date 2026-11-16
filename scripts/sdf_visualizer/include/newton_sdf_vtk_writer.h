// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#  if defined(NEWTON_SDF_VTK_WRITER_EXPORTS)
#    define NSV_API __declspec(dllexport)
#  else
#    define NSV_API __declspec(dllimport)
#  endif
#else
#  define NSV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// All arrays are contiguous and little-endian. Return 0 on success and -1 on error.
// On failure, a human-readable message is copied into error_buffer when provided.
NSV_API int nsv_write_vtp(
    const char* path_utf8,
    const float* points_xyz,
    std::uint64_t point_count,
    const std::int64_t* triangles,
    std::uint64_t triangle_count,
    char* error_buffer,
    std::size_t error_capacity);

NSV_API int nsv_write_vti(
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
    std::size_t error_capacity);

NSV_API int nsv_write_vtm(
    const char* path_utf8,
    const char* const* dataset_names_utf8,
    const char* const* relative_files_utf8,
    std::uint64_t dataset_count,
    char* error_buffer,
    std::size_t error_capacity);

#ifdef __cplusplus
}  // extern "C"
#endif
