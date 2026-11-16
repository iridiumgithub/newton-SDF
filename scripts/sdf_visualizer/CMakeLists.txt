# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(newton_sdf_vtk_writer VERSION 0.1.0 LANGUAGES CXX)

option(NEWTON_SDF_VTK_WRITER_BUILD_TESTS "Build native VTK writer smoke tests" ON)
option(NEWTON_SDF_VTK_WRITER_ENABLE_CUDA "Enable direct CUDA device-pointer VTI export when the CUDA Toolkit is available" ON)

if(NEWTON_SDF_VTK_WRITER_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()

add_library(newton_sdf_vtk_writer SHARED
    src/vtk_writer.cpp
)

target_compile_features(newton_sdf_vtk_writer PUBLIC cxx_std_17)
target_include_directories(newton_sdf_vtk_writer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(newton_sdf_vtk_writer PRIVATE NEWTON_SDF_VTK_WRITER_EXPORTS)
if(NEWTON_SDF_VTK_WRITER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(newton_sdf_vtk_writer PRIVATE NSV_WITH_CUDA=1)
    target_link_libraries(newton_sdf_vtk_writer PRIVATE CUDA::cudart)
    message(STATUS "Native SDF VTK writer: CUDA device-pointer export enabled")
else()
    message(STATUS "Native SDF VTK writer: CUDA Toolkit not found; host-buffer export only")
endif()
set_target_properties(newton_sdf_vtk_writer PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE YES
)

if(MSVC)
    target_compile_options(newton_sdf_vtk_writer PRIVATE /W4 /permissive-)
else()
    target_compile_options(newton_sdf_vtk_writer PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)
install(TARGETS newton_sdf_vtk_writer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/newton_sdf_vtk_writer.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(NEWTON_SDF_VTK_WRITER_BUILD_TESTS)
    enable_testing()
    add_executable(test_vtk_writer tests/test_vtk_writer.cpp)
    target_compile_features(test_vtk_writer PRIVATE cxx_std_17)
    target_link_libraries(test_vtk_writer PRIVATE newton_sdf_vtk_writer)
    add_test(
        NAME newton_sdf_vtk_writer_smoke
        COMMAND test_vtk_writer "${CMAKE_CURRENT_BINARY_DIR}/test-output"
    )

    if(NEWTON_SDF_VTK_WRITER_ENABLE_CUDA AND CUDAToolkit_FOUND)
        add_executable(test_vtk_writer_cuda tests/test_vtk_writer_cuda.cpp)
        target_compile_features(test_vtk_writer_cuda PRIVATE cxx_std_17)
        target_link_libraries(test_vtk_writer_cuda PRIVATE newton_sdf_vtk_writer CUDA::cudart)
        add_test(
            NAME newton_sdf_vtk_writer_cuda
            COMMAND test_vtk_writer_cuda "${CMAKE_CURRENT_BINARY_DIR}/cuda-test-output"
        )
    endif()
endif()
