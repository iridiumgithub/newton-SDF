// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include "newton_sdf_vtk_writer.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool contains_text(const std::filesystem::path& path, const std::string& needle) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    const std::string content(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>());
    return content.find(needle) != std::string::npos;
}

void check(int result, const char* error, const char* operation) {
    if (result != 0) {
        std::cerr << operation << " failed: " << error << "\n";
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path output_dir =
        argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path() / "newton-sdf-vtk-test";
    std::filesystem::create_directories(output_dir);

    const std::vector<float> points{
        0.0F, 0.0F, 0.0F,
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
    };
    const std::vector<std::int64_t> triangles{0, 1, 2};
    const std::vector<float> values{
        -1.0F, 1.0F,
        -0.5F, 0.5F,
        -0.25F, 0.25F,
        -0.1F, 0.1F,
    };
    const double origin[3]{0.0, 0.0, 0.0};
    const double spacing[3]{1.0, 1.0, 1.0};
    char error[1024]{};

    const auto mesh_path = output_dir / "triangle.vtp";
    const auto volume_path = output_dir / "volume.vti";
    const auto multiblock_path = output_dir / "case.vtm";

    check(
        nsv_write_vtp(
            mesh_path.u8string().c_str(),
            points.data(),
            3,
            triangles.data(),
            1,
            error,
            sizeof(error)),
        error,
        "nsv_write_vtp");

    check(
        nsv_write_vti(
            volume_path.u8string().c_str(),
            2,
            2,
            2,
            origin,
            spacing,
            values.data(),
            values.size(),
            "signed_distance_m",
            error,
            sizeof(error)),
        error,
        "nsv_write_vti");

    const char* names[]{"Triangle mesh", "SDF volume"};
    const char* files[]{"triangle.vtp", "volume.vti"};
    check(
        nsv_write_vtm(
            multiblock_path.u8string().c_str(),
            names,
            files,
            2,
            error,
            sizeof(error)),
        error,
        "nsv_write_vtm");

    for (const auto& path : {mesh_path, volume_path, multiblock_path}) {
        if (!std::filesystem::exists(path) || std::filesystem::file_size(path) == 0) {
            std::cerr << "Expected a non-empty output file: " << path << "\n";
            return 1;
        }
    }
    if (!contains_text(mesh_path, "format=\"appended\"") ||
        !contains_text(volume_path, "signed_distance_m") ||
        !contains_text(multiblock_path, "Triangle mesh") ||
        !contains_text(multiblock_path, "volume.vti")) {
        std::cerr << "Native VTK writer smoke test found missing XML metadata.\n";
        return 1;
    }

    std::cout << "Native VTK writer smoke test passed: " << output_dir << "\n";
    return 0;
}
