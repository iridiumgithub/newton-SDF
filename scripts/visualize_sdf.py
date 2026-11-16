# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a texture SDF from an STL and export ParaView-ready diagnostics.

Quick start:
    cmake -S scripts/sdf_visualizer -B scripts/sdf_visualizer/build -DCMAKE_BUILD_TYPE=Release
    cmake --build scripts/sdf_visualizer/build --parallel
    uv sync --extra importers
    python scripts/visualize_sdf.py --input cases/hollow_cylinder.stl --output outputs/hollow_cylinder

VTK XML serialization is handled by the native C++17 writer. The SDF field is
sampled on the CUDA device by Newton's Warp kernel before the data is passed
to the writer as contiguous buffers. When the native library is built with the CUDA Toolkit,
it receives the Warp device pointer and performs bounded, asynchronous D2H transfers.

Open outputs/hollow_cylinder/case.vtm in ParaView. The multiblock contains the
source mesh, Newton's extracted zero isosurface, a sampled 3-D SDF volume, and
three center-plane slices. The volume and slices contain the actual interpolated
Newton texture-SDF values, not values recomputed from the source STL.

STL files do not encode physical units. The default input unit is metres; pass
--input-unit mm (or cm/inch) when the STL coordinates use another unit.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData, _combine_dual_sdf_values, texture_sample_sdf


@wp.kernel
def _sample_texture_sdf_grid_kernel(
    sdf_data: wp.array(dtype=TextureSDFData),
    origin: wp.vec3,
    spacing: wp.vec3,
    dims: wp.vec3i,
    # output
    values: wp.array(dtype=wp.float32),
):
    """Sample the interpolated texture SDF on an axis-aligned ImageData grid."""
    tid = wp.tid()
    nx = dims[0]
    ny = dims[1]

    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    position = origin + wp.vec3(
        float(ix) * spacing[0],
        float(iy) * spacing[1],
        float(iz) * spacing[2],
    )
    values[tid] = texture_sample_sdf(sdf_data[0], position)


@wp.kernel
def _sample_texture_sdf_dual_grid_kernel(
    sdf_a_data: wp.array(dtype=TextureSDFData),
    sdf_b_data: wp.array(dtype=TextureSDFData),
    origin: wp.vec3,
    spacing: wp.vec3,
    dims: wp.vec3i,
    values_a: wp.array(dtype=wp.float32),
    values_b: wp.array(dtype=wp.float32),
    values_combined: wp.array(dtype=wp.float32),
):
    """Sample two fields independently, then compose their interpolated values."""
    tid = wp.tid()
    nx = dims[0]
    ny = dims[1]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    position = origin + wp.vec3(
        float(ix) * spacing[0],
        float(iy) * spacing[1],
        float(iz) * spacing[2],
    )

    distance_a = texture_sample_sdf(sdf_a_data[0], position)
    distance_b = texture_sample_sdf(sdf_b_data[0], position)
    values_a[tid] = distance_a
    values_b[tid] = distance_b
    # Negative-inside intersection: max(A, B). Compose only after each
    # field has independently performed its texture interpolation.
    values_combined[tid] = _combine_dual_sdf_values(distance_a, distance_b)


@wp.kernel
def _reduce_sdf_stats_kernel(values: wp.array(dtype=wp.float32), stats: wp.array(dtype=wp.float32)):
    """Reduce min, max, and negative-sample count to three device scalars."""
    value = values[wp.tid()]
    wp.atomic_min(stats, 0, value)
    wp.atomic_max(stats, 1, value)
    if value < 0.0:
        wp.atomic_add(stats, 2, 1.0)


_UNIT_TO_METERS = {
    "m": 1.0,
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "inch": 0.0254,
}


def _vec3_to_numpy(value: Any) -> np.ndarray:
    return np.asarray([float(value[i]) for i in range(3)], dtype=np.float64)


class NativeVtkWriter:
    """Thin ctypes binding to the C++17 VTK XML writer shipped with this tool."""

    def __init__(self, library_path: Path | None = None):
        script_dir = Path(__file__).resolve().parent
        build_dir = script_dir / "sdf_visualizer" / "build"
        library_names = (
            ("newton_sdf_vtk_writer.dll", "libnewton_sdf_vtk_writer.dll")
            if sys.platform == "win32"
            else ("libnewton_sdf_vtk_writer.dylib",)
            if sys.platform == "darwin"
            else ("libnewton_sdf_vtk_writer.so",)
        )

        env_library = os.environ.get("NEWTON_SDF_VTK_WRITER_LIBRARY")
        if library_path is not None:
            candidates = [library_path.expanduser().resolve()]
        elif env_library:
            candidates = [Path(env_library).expanduser().resolve()]
        else:
            candidates = [
                candidate
                for name in library_names
                for candidate in (
                    build_dir / name,
                    build_dir / "Release" / name,
                    build_dir / "Debug" / name,
                )
            ]

        resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
        if resolved is None:
            expected = "\n".join(f"  - {candidate}" for candidate in candidates)
            raise RuntimeError(
                "Native C++ VTK writer was not found. Build it first:\n"
                "  cmake -S scripts/sdf_visualizer -B scripts/sdf_visualizer/build "
                "-DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build scripts/sdf_visualizer/build --config Release --parallel\n"
                f"Looked for:\n{expected}\n"
                "Alternatively set NEWTON_SDF_VTK_WRITER_LIBRARY to the compiled library path."
            )

        try:
            self._library = ctypes.CDLL(str(resolved))
        except OSError as exc:
            raise RuntimeError(f"Cannot load native VTK writer library {resolved}: {exc}") from exc
        self.library_path = resolved

        error_pointer = ctypes.POINTER(ctypes.c_char)
        try:
            self._library.nsv_has_cuda_support.argtypes = []
            self._library.nsv_has_cuda_support.restype = ctypes.c_int
            self.cuda_enabled = bool(self._library.nsv_has_cuda_support())

            self._library.nsv_write_vti_cuda.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_char_p,
                error_pointer,
                ctypes.c_size_t,
            ]
            self._library.nsv_write_vti_cuda.restype = ctypes.c_int
        except AttributeError as exc:
            raise RuntimeError(
                f"Native VTK writer {resolved} is outdated. Rebuild scripts/sdf_visualizer "
                "after updating the repository."
            ) from exc
        self._library.nsv_write_vtp.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            error_pointer,
            ctypes.c_size_t,
        ]
        self._library.nsv_write_vtp.restype = ctypes.c_int

        self._library.nsv_write_vti.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            error_pointer,
            ctypes.c_size_t,
        ]
        self._library.nsv_write_vti.restype = ctypes.c_int

        self._library.nsv_write_vtm.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint64,
            error_pointer,
            ctypes.c_size_t,
        ]
        self._library.nsv_write_vtm.restype = ctypes.c_int

    @staticmethod
    def _path_bytes(path: Path) -> bytes:
        return str(path.expanduser().resolve()).encode("utf-8")

    @staticmethod
    def _check(result: int, error: ctypes.Array) -> None:
        if result != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"Native VTK writer failed: {message or 'unknown C++ error'}")

    def write_vtp(self, path: Path, points: np.ndarray, faces: np.ndarray) -> None:
        # 仅在必要时整理为 C 连续、小端数组；C++ 直接写入这些缓冲区。
        points_array = np.ascontiguousarray(points, dtype=np.dtype("<f4")).reshape(-1, 3)
        faces_array = np.ascontiguousarray(faces, dtype=np.dtype("<i8")).reshape(-1, 3)
        error = ctypes.create_string_buffer(2048)
        result = self._library.nsv_write_vtp(
            self._path_bytes(path),
            points_array.ctypes.data_as(ctypes.c_void_p),
            len(points_array),
            faces_array.ctypes.data_as(ctypes.c_void_p),
            len(faces_array),
            error,
            len(error),
        )
        self._check(result, error)

    def write_vti(
        self,
        path: Path,
        dims: tuple[int, int, int],
        origin: np.ndarray,
        spacing: np.ndarray,
        values: np.ndarray,
        array_name: str = "signed_distance_m",
    ) -> None:
        nx, ny, nz = (int(value) for value in dims)
        expected_count = nx * ny * nz
        values_array = np.ascontiguousarray(values, dtype=np.dtype("<f4")).reshape(-1)
        if values_array.size != expected_count:
            raise ValueError(
                f"VTI sample count mismatch: expected {expected_count}, got {values_array.size}."
            )
        origin_array = np.ascontiguousarray(origin, dtype=np.float64).reshape(3)
        spacing_array = np.ascontiguousarray(spacing, dtype=np.float64).reshape(3)
        error = ctypes.create_string_buffer(2048)
        result = self._library.nsv_write_vti(
            self._path_bytes(path),
            nx,
            ny,
            nz,
            origin_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            spacing_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            values_array.ctypes.data_as(ctypes.c_void_p),
            values_array.size,
            array_name.encode("utf-8"),
            error,
            len(error),
        )
        self._check(result, error)

    def write_vti_cuda(
        self,
        path: Path,
        dims: tuple[int, int, int],
        origin: np.ndarray,
        spacing: np.ndarray,
        values: wp.array,
        device: wp.Device,
        array_name: str = "signed_distance_m",
    ) -> bool:
        """Write a contiguous Warp float32 device array without calling array.numpy()."""
        if not self.cuda_enabled:
            return False
        if not device.is_cuda or not values.device.is_cuda or values.device.ordinal != device.ordinal:
            raise ValueError("CUDA VTI export requires a Warp array on the selected CUDA device.")
        if values.dtype != wp.float32 or not values.is_contiguous or values.ndim != 1:
            raise ValueError("CUDA VTI export requires a contiguous one-dimensional Warp float32 array.")

        nx, ny, nz = (int(value) for value in dims)
        expected_count = nx * ny * nz
        if values.size != expected_count:
            raise ValueError(f"VTI sample count mismatch: expected {expected_count}, got {values.size}.")

        origin_array = np.ascontiguousarray(origin, dtype=np.float64).reshape(3)
        spacing_array = np.ascontiguousarray(spacing, dtype=np.float64).reshape(3)
        if not device.is_primary or not values.device.is_primary:
            # CUDA Runtime stream interop requires Warp's primary CUDA context.
            return False
        warp_stream = wp.get_stream(device)
        raw_stream = warp_stream.cuda_stream
        stream_value = raw_stream.value if isinstance(raw_stream, ctypes.c_void_p) else raw_stream
        stream_handle = int(stream_value or 0)
        error = ctypes.create_string_buffer(2048)
        result = self._library.nsv_write_vti_cuda(
            self._path_bytes(path),
            nx,
            ny,
            nz,
            origin_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            spacing_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_void_p(int(values.ptr)),
            values.size,
            int(device.ordinal),
            ctypes.c_void_p(stream_handle),
            array_name.encode("utf-8"),
            error,
            len(error),
        )
        self._check(result, error)
        return True

    def write_vtm(self, path: Path, datasets: list[tuple[str, Path]], output_dir: Path) -> None:
        names = [name.encode("utf-8") for name, _ in datasets]
        files = [dataset_path.relative_to(output_dir).as_posix().encode("utf-8") for _, dataset_path in datasets]
        names_array = (ctypes.c_char_p * len(names))(*names)
        files_array = (ctypes.c_char_p * len(files))(*files)
        error = ctypes.create_string_buffer(2048)
        result = self._library.nsv_write_vtm(
            self._path_bytes(path),
            names_array,
            files_array,
            len(datasets),
            error,
            len(error),
        )
        self._check(result, error)


def _write_vtp(writer: NativeVtkWriter, path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    """Write triangle PolyData through the native C++ raw-appended VTK writer."""
    writer.write_vtp(path, points, faces)


def _write_vti(
    writer: NativeVtkWriter,
    path: Path,
    dims: tuple[int, int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
    values: wp.array | np.ndarray,
    device: wp.Device,
    array_name: str = "signed_distance_m",
) -> None:
    """Write ImageData; use the Warp device pointer when CUDA Runtime support is available."""
    if isinstance(values, wp.array):
        if writer.write_vti_cuda(path, dims, origin, spacing, values, device, array_name):
            return
        # Compatibility fallback for native libraries built without the CUDA Toolkit.
        values = values.numpy()
    writer.write_vti(path, dims, origin, spacing, values, array_name)


def _write_vtm(
    writer: NativeVtkWriter,
    path: Path,
    datasets: list[tuple[str, Path]],
    output_dir: Path,
) -> None:
    """Write the ParaView multiblock index through the native C++ writer."""
    writer.write_vtm(path, datasets, output_dir)


def _grid_dims(extents: np.ndarray, max_resolution: int) -> tuple[int, int, int]:
    """Choose an aspect-ratio-preserving grid with max_resolution on the longest axis."""
    longest = float(np.max(extents))
    if longest <= 0.0:
        raise ValueError("SDF bounds must have positive extent on all axes.")
    ratios = extents / longest
    dims = np.maximum(2, np.ceil(ratios * (max_resolution - 1)).astype(np.int64) + 1)
    return tuple(int(v) for v in dims)


def _fit_volume_dims(
    extents: np.ndarray,
    requested_resolution: int,
    max_points: int,
) -> tuple[int, int, int]:
    """Reduce visualization resolution when a full 3-D volume would be too large."""
    if requested_resolution < 2:
        raise ValueError("Visualization resolution must be at least 2.")
    if max_points < 8:
        raise ValueError("Maximum sample count must be at least 8.")

    low, high = 2, requested_resolution
    best = _grid_dims(extents, low)
    while low <= high:
        mid = (low + high) // 2
        dims = _grid_dims(extents, mid)
        if math.prod(dims) <= max_points:
            best = dims
            low = mid + 1
        else:
            high = mid - 1
    return best


def _fit_slice_dims(
    extents: np.ndarray,
    axis_u: int,
    axis_v: int,
    requested_resolution: int,
    max_points: int,
) -> tuple[int, int]:
    """Choose 2-D slice dimensions while respecting a point-count limit."""
    slice_extents = np.asarray([extents[axis_u], extents[axis_v]], dtype=np.float64)
    longest = float(np.max(slice_extents))
    if longest <= 0.0:
        raise ValueError("Slice axes must have positive extent.")

    low, high = 2, requested_resolution
    if max_points < 4:
        raise ValueError("Maximum slice sample count must be at least 4.")
    best = (2, 2)
    while low <= high:
        mid = (low + high) // 2
        dims = np.maximum(2, np.ceil(slice_extents / longest * (mid - 1)).astype(np.int64) + 1)
        candidate = (int(dims[0]), int(dims[1]))
        if candidate[0] * candidate[1] <= max_points:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _sample_sdf_grid(
    sdf_data_array: wp.array,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: tuple[int, int, int],
    device: wp.Device,
) -> wp.array:
    """Sample the actual Newton texture SDF and retain the result on the GPU."""
    total = math.prod(dims)
    output = wp.empty(total, dtype=wp.float32, device=device)
    wp.launch(
        _sample_texture_sdf_grid_kernel,
        dim=total,
        inputs=[
            sdf_data_array,
            wp.vec3(float(origin[0]), float(origin[1]), float(origin[2])),
            wp.vec3(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            wp.vec3i(int(dims[0]), int(dims[1]), int(dims[2])),
        ],
        outputs=[output],
        device=device,
    )
    return output


def _sample_dual_sdf_grid(
    sdf_a_data: wp.array,
    sdf_b_data: wp.array,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: tuple[int, int, int],
    device: wp.Device,
) -> tuple[wp.array, wp.array, wp.array]:
    """Sample A and B independently, then compute max(A, B), entirely on CUDA."""
    total = math.prod(dims)
    values_a = wp.empty(total, dtype=wp.float32, device=device)
    values_b = wp.empty(total, dtype=wp.float32, device=device)
    values_combined = wp.empty(total, dtype=wp.float32, device=device)
    wp.launch(
        _sample_texture_sdf_dual_grid_kernel,
        dim=total,
        inputs=[
            sdf_a_data,
            sdf_b_data,
            wp.vec3(float(origin[0]), float(origin[1]), float(origin[2])),
            wp.vec3(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            wp.vec3i(int(dims[0]), int(dims[1]), int(dims[2])),
        ],
        outputs=[values_a, values_b, values_combined],
        device=device,
    )
    return values_a, values_b, values_combined


def _sdf_grid_stats(values: wp.array, device: wp.Device) -> dict[str, float]:
    """Copy only three reduced scalars to the CPU, never the full sample field."""
    if values.size == 0:
        raise ValueError("Cannot compute statistics for an empty SDF sample array.")
    stats = wp.array(
        np.asarray([np.inf, -np.inf, 0.0], dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
    wp.launch(_reduce_sdf_stats_kernel, dim=values.size, inputs=[values], outputs=[stats], device=device)
    minimum, maximum, negative_count = (float(value) for value in stats.numpy())
    return {
        "minimum_m": minimum,
        "maximum_m": maximum,
        "negative_sample_fraction": negative_count / values.size,
    }


def _load_stl(path: Path, unit: str) -> tuple[newton.Mesh, dict[str, Any]]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "STL loading requires trimesh. Install the importer dependencies with: uv sync --extra importers"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"STL input file does not exist: {path}")
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Expected an STL file, got: {path}")

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No triangle mesh geometry found in STL input: {path}")
        loaded = trimesh.util.concatenate(geometries)

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported STL loader result: {type(loaded).__name__}")

    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("STL must contain at least four 3-D vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError("STL must contain at least four triangles.")
    if not np.isfinite(vertices).all():
        raise ValueError("STL contains NaN or infinite vertex coordinates.")

    scale = _UNIT_TO_METERS[unit]
    vertices *= np.float32(scale)
    mesh = newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False)

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    info = {
        "input_file": str(path.resolve()),
        "input_unit": unit,
        "unit_to_meter_scale": scale,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "watertight": bool(mesh.is_watertight),
        "bounds_min_m": bounds_min.tolist(),
        "bounds_max_m": bounds_max.tolist(),
        "bounds_extent_m": (bounds_max - bounds_min).tolist(),
    }
    return mesh, info


def _export_sdf_diagnostics(
    writer: NativeVtkWriter,
    sdf: newton.SDF,
    output_dir: Path,
    device: wp.Device,
    volume_resolution: int,
    slice_resolution: int,
    max_volume_points: int,
    max_slice_points: int,
    export_volume: bool,
    export_slices: bool,
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    texture_data = sdf.to_texture_kernel_data()
    if texture_data is None:
        raise RuntimeError("SDF build returned no texture data; cannot export texture samples.")

    lower = _vec3_to_numpy(texture_data.sdf_box_lower)
    upper = _vec3_to_numpy(texture_data.sdf_box_upper)
    extents = upper - lower
    if np.any(extents <= 0.0):
        raise RuntimeError(f"Invalid SDF bounds: lower={lower.tolist()}, upper={upper.tolist()}")

    sdf_data_array = wp.array([texture_data], dtype=TextureSDFData, device=device)
    datasets: list[tuple[str, Path]] = []
    sample_info: dict[str, Any] = {
        "sdf_bounds_min_m": lower.tolist(),
        "sdf_bounds_max_m": upper.tolist(),
        "sdf_bounds_extent_m": extents.tolist(),
        "voxel_size_m": _vec3_to_numpy(texture_data.voxel_size).tolist(),
        "subgrid_size": int(texture_data.subgrid_size),
        "num_subgrids": int(texture_data.num_subgrids),
        "paired_samples": bool(texture_data.paired_samples),
        "sampling_backend": "Warp CUDA kernel calling Newton texture_sample_sdf",
        "sampled_datasets": {},
    }

    if export_volume:
        dims = _fit_volume_dims(extents, volume_resolution, max_volume_points)
        spacing = extents / (np.asarray(dims, dtype=np.float64) - 1.0)
        values = _sample_sdf_grid(sdf_data_array, lower, spacing, dims, device)
        path = output_dir / "sdf_volume.vti"
        _write_vti(writer, path, dims, lower, spacing, values, device)
        datasets.append(("Sampled SDF volume", path))
        sample_info["sampled_datasets"]["sdf_volume"] = {
            "file": path.name,
            "dimensions": list(dims),
            "spacing_m": spacing.tolist(),
            "sample_count": int(values.size),
            **_sdf_grid_stats(values, device),
        }
        print(f"  Exported volume: {path} ({dims[0]} x {dims[1]} x {dims[2]})")

    if export_slices:
        slices_dir = output_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        center = 0.5 * (lower + upper)
        slice_specs = [
            ("xy", 0, 1, 2, "XY center slice"),
            ("xz", 0, 2, 1, "XZ center slice"),
            ("yz", 1, 2, 0, "YZ center slice"),
        ]
        for suffix, axis_u, axis_v, fixed_axis, display_name in slice_specs:
            nu, nv = _fit_slice_dims(
                extents,
                axis_u,
                axis_v,
                slice_resolution,
                max_slice_points,
            )
            dims_list = [1, 1, 1]
            dims_list[axis_u] = nu
            dims_list[axis_v] = nv
            dims = tuple(dims_list)

            origin = lower.copy()
            origin[fixed_axis] = center[fixed_axis]
            spacing = np.ones(3, dtype=np.float64)
            spacing[axis_u] = extents[axis_u] / (nu - 1)
            spacing[axis_v] = extents[axis_v] / (nv - 1)

            values = _sample_sdf_grid(sdf_data_array, origin, spacing, dims, device)
            path = slices_dir / f"sdf_slice_{suffix}.vti"
            _write_vti(writer, path, dims, origin, spacing, values, device)
            datasets.append((display_name, path))
            sample_info["sampled_datasets"][f"slice_{suffix}"] = {
                "file": path.relative_to(output_dir).as_posix(),
                "dimensions": list(dims),
                "origin_m": origin.tolist(),
                "spacing_m": spacing.tolist(),
                "sample_count": int(values.size),
                **_sdf_grid_stats(values, device),
            }
            print(f"  Exported {suffix.upper()} slice: {path} ({nu} x {nv})")

    return datasets, sample_info


def _export_dual_sdf_diagnostics(
    writer: NativeVtkWriter,
    sdf_a: newton.SDF,
    sdf_b: newton.SDF,
    output_dir: Path,
    device: wp.Device,
    volume_resolution: int,
    slice_resolution: int,
    max_volume_points: int,
    max_slice_points: int,
    export_volume: bool,
    export_slices: bool,
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """Export A, B, and max(A, B) sampled after independent texture interpolation."""
    texture_a = sdf_a.to_texture_kernel_data()
    texture_b = sdf_b.to_texture_kernel_data()
    if texture_a is None or texture_b is None:
        raise RuntimeError("Both dual-field SDFs must contain texture data.")

    lower_a = _vec3_to_numpy(texture_a.sdf_box_lower)
    upper_a = _vec3_to_numpy(texture_a.sdf_box_upper)
    lower_b = _vec3_to_numpy(texture_b.sdf_box_lower)
    upper_b = _vec3_to_numpy(texture_b.sdf_box_upper)
    # Both texture samplers extrapolate positively outside their own AABBs.
    # Restrict the common grid to the overlap so max(A, B) never relies on
    # out-of-domain extrapolation from either independently baked field.
    lower = np.maximum(lower_a, lower_b)
    upper = np.minimum(upper_a, upper_b)
    extents = upper - lower
    if np.any(extents <= 0.0):
        raise RuntimeError(
            "The two SDF texture domains do not overlap. Increase --margin (and, if needed, "
            "--narrow-band) so both independently baked fields cover the intended intersection region. "
            f"A=[{lower_a.tolist()}, {upper_a.tolist()}], B=[{lower_b.tolist()}, {upper_b.tolist()}]."
        )

    sdf_a_data = wp.array([texture_a], dtype=TextureSDFData, device=device)
    sdf_b_data = wp.array([texture_b], dtype=TextureSDFData, device=device)
    datasets: list[tuple[str, Path]] = []
    sample_info: dict[str, Any] = {
        "composition": "max(field_a, field_b)",
        "inside_convention": "negative",
        "distance_metric": "CSG max composition; the sign and zero set are meaningful, but the result is not guaranteed to be an exact Euclidean SDF near non-smooth intersections",
        "sdf_bounds_min_m": lower.tolist(),
        "sdf_bounds_max_m": upper.tolist(),
        "sdf_bounds_extent_m": extents.tolist(),
        "sampling_backend": "Warp CUDA kernel; interpolate A and B independently before max composition",
        "fields": {
            "a": {
                "sdf_bounds_min_m": lower_a.tolist(),
                "sdf_bounds_max_m": upper_a.tolist(),
                "voxel_size_m": _vec3_to_numpy(texture_a.voxel_size).tolist(),
                "subgrid_size": int(texture_a.subgrid_size),
                "num_subgrids": int(texture_a.num_subgrids),
                "paired_samples": bool(texture_a.paired_samples),
            },
            "b": {
                "sdf_bounds_min_m": lower_b.tolist(),
                "sdf_bounds_max_m": upper_b.tolist(),
                "voxel_size_m": _vec3_to_numpy(texture_b.voxel_size).tolist(),
                "subgrid_size": int(texture_b.subgrid_size),
                "num_subgrids": int(texture_b.num_subgrids),
                "paired_samples": bool(texture_b.paired_samples),
            },
        },
        "sampled_datasets": {},
    }

    def export_triplet(
        suffix: str,
        dims: tuple[int, int, int],
        origin: np.ndarray,
        spacing: np.ndarray,
        names: tuple[str, str, str],
        labels: tuple[str, str, str],
    ) -> None:
        values_a, values_b, values_combined = _sample_dual_sdf_grid(
            sdf_a_data, sdf_b_data, origin, spacing, dims, device
        )
        for field_key, display_name, file_name, values in (
            ("field_a", labels[0], names[0], values_a),
            ("field_b", labels[1], names[1], values_b),
            ("combined", labels[2], names[2], values_combined),
        ):
            path = output_dir / file_name
            _write_vti(writer, path, dims, origin, spacing, values, device)
            datasets.append((display_name, path))
            sample_info["sampled_datasets"][f"{field_key}_{suffix}"] = {
                "file": path.relative_to(output_dir).as_posix(),
                "dimensions": list(dims),
                "origin_m": origin.tolist(),
                "spacing_m": spacing.tolist(),
                "sample_count": int(values.size),
                **_sdf_grid_stats(values, device),
            }
            print(f"  Exported {display_name}: {path}")

    if export_volume:
        dims = _fit_volume_dims(extents, volume_resolution, max_volume_points)
        spacing = extents / (np.asarray(dims, dtype=np.float64) - 1.0)
        export_triplet(
            "volume",
            dims,
            lower,
            spacing,
            ("sdf_field_a.vti", "sdf_field_b.vti", "sdf_combined.vti"),
            ("SDF field A volume", "SDF field B volume", "Combined max(A, B) volume"),
        )

    if export_slices:
        slices_dir = output_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        center = 0.5 * (lower + upper)
        slice_specs = (
            ("xy", 0, 1, 2),
            ("xz", 0, 2, 1),
            ("yz", 1, 2, 0),
        )
        for suffix, axis_u, axis_v, fixed_axis in slice_specs:
            nu, nv = _fit_slice_dims(extents, axis_u, axis_v, slice_resolution, max_slice_points)
            dims_list = [1, 1, 1]
            dims_list[axis_u] = nu
            dims_list[axis_v] = nv
            dims = tuple(dims_list)

            origin = lower.copy()
            origin[fixed_axis] = center[fixed_axis]
            spacing = np.ones(3, dtype=np.float64)
            spacing[axis_u] = extents[axis_u] / (nu - 1)
            spacing[axis_v] = extents[axis_v] / (nv - 1)
            export_triplet(
                f"slice_{suffix}",
                dims,
                origin,
                spacing,
                (
                    f"slices/sdf_field_a_slice_{suffix}.vti",
                    f"slices/sdf_field_b_slice_{suffix}.vti",
                    f"slices/sdf_combined_slice_{suffix}.vti",
                ),
                (
                    f"SDF field A {suffix.upper()} slice",
                    f"SDF field B {suffix.upper()} slice",
                    f"Combined max(A, B) {suffix.upper()} slice",
                ),
            )

    return datasets, sample_info


def _texture_layout_info(sdf: newton.SDF) -> dict[str, Any]:
    texture_data = sdf.to_texture_kernel_data()
    if texture_data is None:
        raise RuntimeError("SDF has no texture data.")
    coarse_texture = sdf._coarse_texture
    subgrid_texture = sdf._subgrid_texture
    coarse_dimensions = (
        [int(coarse_texture.width), int(coarse_texture.height), int(coarse_texture.depth)]
        if coarse_texture is not None
        else None
    )
    coarse_cells = [dimension - 1 for dimension in coarse_dimensions] if coarse_dimensions is not None else None
    subgrid_size = int(texture_data.subgrid_size)
    fine_cells = [dimension * subgrid_size for dimension in coarse_cells] if coarse_cells is not None else None
    return {
        "coarse_texture_dimensions": coarse_dimensions,
        "coarse_cell_dimensions": coarse_cells,
        "fine_grid_cell_dimensions": fine_cells,
        "fine_grid_sample_dimensions": [dimension + 1 for dimension in fine_cells] if fine_cells is not None else None,
        "subgrid_texture_dimensions": (
            [int(subgrid_texture.width), int(subgrid_texture.height), int(subgrid_texture.depth)]
            if subgrid_texture is not None
            else None
        ),
        "subgrid_size": subgrid_size,
        "num_subgrids": int(texture_data.num_subgrids),
        "paired_samples": bool(texture_data.paired_samples),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Newton texture SDF from an STL and export ParaView diagnostics with the native C++ writer."
    )
    parser.add_argument("--input", "--case", dest="input", type=Path, required=True, help="Input STL case file.")
    parser.add_argument(
        "--input-b",
        type=Path,
        default=None,
        help="Optional second STL for dual-field intersection visualization; both STL files must share one world frame.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--native-writer",
        type=Path,
        default=None,
        help="Path to the compiled C++ VTK writer library (or use NEWTON_SDF_VTK_WRITER_LIBRARY).",
    )
    parser.add_argument(
        "--input-unit",
        choices=tuple(_UNIT_TO_METERS),
        default="m",
        help="Physical unit used by STL coordinates. STL has no unit metadata (default: m).",
    )

    resolution_group = parser.add_mutually_exclusive_group()
    resolution_group.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Maximum SDF grid resolution along the longest axis (default: 128).",
    )
    resolution_group.add_argument(
        "--target-voxel-size",
        type=float,
        default=None,
        help="Target SDF voxel size in metres; takes precedence over maximum resolution.",
    )

    parser.add_argument("--margin", type=float, default=0.02, help="SDF AABB padding in metres (default: 0.02).")
    parser.add_argument(
        "--narrow-band",
        type=float,
        default=0.05,
        help="Symmetric narrow-band half-width in metres (default: 0.05).",
    )
    parser.add_argument(
        "--sign-method",
        choices=("auto", "parity", "winding", "normal"),
        default="auto",
        help="Inside/outside sign strategy (default: auto).",
    )
    parser.add_argument(
        "--texture-format",
        choices=("float32", "uint16", "uint8"),
        default="uint16",
        help="Fine texture quantization format (default: uint16).",
    )
    parser.add_argument("--device", default="cuda:0", help="Warp CUDA device (default: cuda:0).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cooked-SDF cache directory.")
    parser.add_argument("--isovalue", type=float, default=0.0, help="Isosurface value to extract in metres.")

    parser.add_argument(
        "--volume-resolution",
        type=int,
        default=96,
        help="Longest-axis resolution for the full 3-D visualization volume (default: 96).",
    )
    parser.add_argument(
        "--slice-resolution",
        type=int,
        default=512,
        help="Longest-axis resolution for each center-plane slice (default: 512).",
    )
    parser.add_argument(
        "--max-volume-points",
        type=int,
        default=2_000_000,
        help="Maximum points in the full 3-D visualization volume (default: 2000000).",
    )
    parser.add_argument(
        "--max-slice-points",
        type=int,
        default=1_000_000,
        help="Maximum points per 2-D slice (default: 1000000).",
    )
    parser.add_argument("--no-volume", action="store_true", help="Skip the full 3-D sampled volume.")
    parser.add_argument("--no-slices", action="store_true", help="Skip the three center-plane slices.")
    parser.add_argument(
        "--no-paired-samples",
        action="store_true",
        help="Disable paired-X texture samples (useful for debugging layout differences).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.resolution is None and args.target_voxel_size is None:
        args.resolution = 128
    if args.resolution is not None and args.resolution < 8:
        parser.error("--resolution must be at least 8.")
    if args.target_voxel_size is not None and args.target_voxel_size <= 0.0:
        parser.error("--target-voxel-size must be positive.")
    if args.margin < 0.0:
        parser.error("--margin must be non-negative.")
    if args.narrow_band <= 0.0:
        parser.error("--narrow-band must be positive.")
    if args.volume_resolution < 2 or args.slice_resolution < 2:
        parser.error("Visualization resolutions must be at least 2.")
    if args.max_volume_points < 8:
        parser.error("--max-volume-points must be at least 8.")
    if args.max_slice_points < 4:
        parser.error("--max-slice-points must be at least 4.")

    # Fail before loading/building the SDF if the native exporter is unavailable.
    writer = NativeVtkWriter(args.native_writer)
    print(f"Native VTK writer: {writer.library_path}")

    if not wp.is_cuda_available():
        raise RuntimeError(
            "Newton texture SDF construction requires CUDA. No CUDA-capable Warp device was detected."
        )
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError(f"Texture SDF construction requires a CUDA device, got {device}.")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading STL: {input_path}")
    mesh, mesh_info = _load_stl(input_path, args.input_unit)
    input_b_path = args.input_b.expanduser().resolve() if args.input_b is not None else None
    mesh_b: newton.Mesh | None = None
    mesh_b_info: dict[str, Any] | None = None
    if input_b_path is not None:
        print(f"  Loading second STL: {input_b_path}")
        mesh_b, mesh_b_info = _load_stl(input_b_path, args.input_unit)
    print(
        f"  Vertices: {mesh_info['vertex_count']}, triangles: {mesh_info['triangle_count']}, "
        f"watertight: {mesh_info['watertight']}"
    )
    for field_label, field_info in (("A", mesh_info), ("B", mesh_b_info)):
        if field_info is None:
            continue
        if not field_info["watertight"] and args.sign_method == "auto":
            print(f"  WARNING: mesh {field_label} is not watertight; auto mode will use winding-number signs.")
        elif not field_info["watertight"] and args.sign_method == "parity":
            print(f"  WARNING: mesh {field_label} is not watertight; parity signs may be unreliable.")

    print(f"[2/5] Building Newton texture SDF on {device} ...")
    build_start = time.perf_counter()
    with wp.ScopedDevice(device):
        sdf = newton.SDF.create_from_mesh(
            mesh,
            device=device,
            max_resolution=args.resolution,
            target_voxel_size=args.target_voxel_size,
            margin=args.margin,
            narrow_band_range=(-args.narrow_band, args.narrow_band),
            texture_format=args.texture_format,
            sign_method=args.sign_method,
            cache_dir=str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None,
            paired_samples=not args.no_paired_samples,
        )
        sdf_b = (
            newton.SDF.create_from_mesh(
                mesh_b,
                device=device,
                max_resolution=args.resolution,
                target_voxel_size=args.target_voxel_size,
                margin=args.margin,
                narrow_band_range=(-args.narrow_band, args.narrow_band),
                texture_format=args.texture_format,
                sign_method=args.sign_method,
                cache_dir=str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None,
                paired_samples=not args.no_paired_samples,
            )
            if mesh_b is not None
            else None
        )
    build_seconds = time.perf_counter() - build_start
    print(f"  SDF build finished in {build_seconds:.3f} s" + (" (A + B)" if sdf_b is not None else ""))

    if sdf.is_empty() or sdf.to_texture_kernel_data() is None:
        raise RuntimeError("SDF A construction produced an empty texture SDF.")
    if sdf_b is not None and (sdf_b.is_empty() or sdf_b.to_texture_kernel_data() is None):
        raise RuntimeError("SDF B construction produced an empty texture SDF.")

    cuda_device_export = writer.cuda_enabled and device.is_primary
    if cuda_device_export:
        print("Native VTI export: direct Warp device pointer + chunked pinned-memory D2H")
    elif writer.cuda_enabled:
        print("WARNING: Warp is using a non-primary CUDA context; VTI export will use the host-copy fallback.")
    else:
        print("WARNING: native library has no CUDA Runtime support; VTI export will use Warp's host-copy fallback.")

    datasets: list[tuple[str, Path]] = []
    source_path = output_dir / "source_mesh.vtp"
    _write_vtp(writer, source_path, mesh.vertices, mesh.indices.reshape(-1, 3))
    datasets.append(("Source STL mesh A", source_path))
    if mesh_b is not None:
        source_b_path = output_dir / "source_mesh_b.vtp"
        _write_vtp(writer, source_b_path, mesh_b.vertices, mesh_b.indices.reshape(-1, 3))
        datasets.append(("Source STL mesh B", source_b_path))
    print(f"[3/5] Exported source mesh(es): {source_path}")

    extraction_start = time.perf_counter()
    zero_mesh = sdf.extract_isomesh(isovalue=args.isovalue, device=device)
    zero_mesh_info: dict[str, Any]
    if zero_mesh is None or len(zero_mesh.indices) == 0:
        zero_mesh_info = {"extracted": False, "vertex_count": 0, "triangle_count": 0}
        print("  WARNING: Newton returned an empty zero isosurface for field A.")
    else:
        zero_path = output_dir / ("newton_zero_isosurface_a.vtp" if sdf_b is not None else "newton_zero_isosurface.vtp")
        zero_faces = np.asarray(zero_mesh.indices, dtype=np.int64).reshape(-1, 3)
        _write_vtp(writer, zero_path, zero_mesh.vertices, zero_faces)
        datasets.append(("Newton zero isosurface A", zero_path))
        zero_mesh_info = {
            "extracted": True,
            "vertex_count": int(len(zero_mesh.vertices)),
            "triangle_count": int(len(zero_faces)),
            "isovalue_m": float(args.isovalue),
        }
        print(f"  Exported field A zero surface: {zero_path}")

    zero_mesh_info_b: dict[str, Any] | None = None
    if sdf_b is not None:
        zero_mesh_b = sdf_b.extract_isomesh(isovalue=args.isovalue, device=device)
        if zero_mesh_b is None or len(zero_mesh_b.indices) == 0:
            zero_mesh_info_b = {"extracted": False, "vertex_count": 0, "triangle_count": 0}
            print("  WARNING: Newton returned an empty zero isosurface for field B.")
        else:
            zero_path_b = output_dir / "newton_zero_isosurface_b.vtp"
            zero_faces_b = np.asarray(zero_mesh_b.indices, dtype=np.int64).reshape(-1, 3)
            _write_vtp(writer, zero_path_b, zero_mesh_b.vertices, zero_faces_b)
            datasets.append(("Newton zero isosurface B", zero_path_b))
            zero_mesh_info_b = {
                "extracted": True,
                "vertex_count": int(len(zero_mesh_b.vertices)),
                "triangle_count": int(len(zero_faces_b)),
                "isovalue_m": float(args.isovalue),
            }
            print(f"  Exported field B zero surface: {zero_path_b}")
    extraction_seconds = time.perf_counter() - extraction_start

    print("[5/5] Sampling interpolated texture SDF field(s) ...")
    sampling_start = time.perf_counter()
    if sdf_b is None:
        sample_datasets, sample_info = _export_sdf_diagnostics(
            writer,
            sdf,
            output_dir,
            device,
            volume_resolution=args.volume_resolution,
            slice_resolution=args.slice_resolution,
            max_volume_points=args.max_volume_points,
            max_slice_points=args.max_slice_points,
            export_volume=not args.no_volume,
            export_slices=not args.no_slices,
        )
    else:
        sample_datasets, sample_info = _export_dual_sdf_diagnostics(
            writer,
            sdf,
            sdf_b,
            output_dir,
            device,
            volume_resolution=args.volume_resolution,
            slice_resolution=args.slice_resolution,
            max_volume_points=args.max_volume_points,
            max_slice_points=args.max_slice_points,
            export_volume=not args.no_volume,
            export_slices=not args.no_slices,
        )
    datasets.extend(sample_datasets)
    sampling_seconds = time.perf_counter() - sampling_start

    multiblock_path = output_dir / "case.vtm"
    _write_vtm(writer, multiblock_path, datasets, output_dir)

    sample_info["texture_layout_a"] = _texture_layout_info(sdf)
    if sdf_b is not None:
        sample_info["texture_layout_b"] = _texture_layout_info(sdf_b)
    else:
        sample_info["texture_layout"] = sample_info.pop("texture_layout_a")

    metadata = {
        "tool": "Newton SDF Inspector",
        "visualization_writer": {
            "backend": "native C++17",
            "library": str(writer.library_path),
            "vtk_binary_encoding": "raw appended binary",
            "cuda_device_pointer_export": cuda_device_export,
        },
        "input": mesh_info,
        "input_b": mesh_b_info,
        "dual_field": {
            "enabled": sdf_b is not None,
            "composition": "max(field_a, field_b)" if sdf_b is not None else None,
            "inside_convention": "negative",
            "distance_metric": (
                "CSG max composition; not guaranteed to be an exact Euclidean SDF near non-smooth intersections"
                if sdf_b is not None else None
            ),
            "coordinate_frame": "shared world frame; no registration is applied",
        },
        "build": {
            "device": str(device),
            "resolution": args.resolution,
            "target_voxel_size_m": args.target_voxel_size,
            "margin_m": args.margin,
            "narrow_band_range_m": [-args.narrow_band, args.narrow_band],
            "sign_method_requested": args.sign_method,
            "sign_method_resolved": (
                "parity" if mesh_info["watertight"] else "winding"
            ) if args.sign_method == "auto" else args.sign_method,
            "sign_method_resolved_b": (
                ("parity" if mesh_b_info["watertight"] else "winding")
                if args.sign_method == "auto" and mesh_b_info is not None
                else args.sign_method if mesh_b_info is not None else None
            ),
            "texture_format": args.texture_format,
            "paired_samples": not args.no_paired_samples,
            "build_seconds": build_seconds,
            "isovalue_m": args.isovalue,
            "field_b_built": sdf_b is not None,
        },
        "extraction": {
            **zero_mesh_info,
            "seconds": extraction_seconds,
            **({"field_a": zero_mesh_info, "field_b": zero_mesh_info_b} if sdf_b is not None else {}),
        },
        "sampling": {
            **sample_info,
            "seconds": sampling_seconds,
        },
        "outputs": [path.relative_to(output_dir).as_posix() for _, path in datasets] + ["case.vtm"],
        "notes": [
            "The VTI values are sampled from Newton's interpolated texture SDF using a Warp CUDA kernel.",
            "VTK XML serialization is performed by the native C++17 writer using raw appended binary arrays (no Base64 encoding).",
            "The Newton zero isosurface is extracted by Newton's texture-SDF Marching Cubes path.",
            "STL has no physical-unit metadata; input coordinates were scaled according to input_unit.",
            (
                "Dual-field mode independently interpolates A and B, then composes max(A, B); the combined sign and zero set are meaningful, but distance magnitudes are not guaranteed to be exact Euclidean distances near non-smooth intersections. Use Contour(0) on sdf_combined.vti for the combined zero surface."
                if sdf_b is not None
                else "Single-field mode exports the Newton texture SDF."
            ),
            "The native CUDA writer copies device samples in bounded pinned-memory chunks on the active Warp stream when built with CUDA Runtime support.",
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_lines = [
        "Newton SDF Inspector output",
        "",
        "VTK files are serialized by the native C++17 writer. Scalar-field sampling runs on the CUDA device through Newton's Warp kernel.",
        "When built with CUDA Runtime support, the writer receives the Warp device pointer and performs chunked D2H copies using pinned host buffers.",
        "",
        "1. Open case.vtm in ParaView to load all exported datasets together.",
        "2. Toggle source meshes and extracted zero isosurfaces to compare geometry.",
    ]
    if not args.no_volume:
        if sdf_b is None:
            readme_lines.extend([
                "3. Select sdf_volume.vti and apply the Contour filter with value 0.0 to extract",
                "   the zero level set from the sampled interpolated SDF.",
            ])
        else:
            readme_lines.extend([
                "3. Compare sdf_field_a.vti and sdf_field_b.vti independently.",
                "4. Select sdf_combined.vti and apply Contour with value 0.0 to inspect max(A, B).",
                "   The two fields are interpolated independently before the intersection composition.",
            ])
    if not args.no_slices:
        readme_lines.extend([
            "5. Open the files under slices/ to inspect signed-distance cross sections.",
            "   Color by signed_distance_m; the material side should be negative for a correctly",
            "   oriented closed solid.",
        ])
    readme_lines.extend([
        "6. Check metadata.json for mesh watertightness, SDF parameters, texture layout,",
        "   sample dimensions, and value ranges.",
        "",
        "For dual-field runs, --input-b must use the same world coordinate frame and units as --input.",
        "The combined field uses max(A, B), which represents intersection when both fields are negative inside.",
        "The full 3-D volume is capped by --max-volume-points. Center slices use",
        "--slice-resolution and --max-slice-points. Increase SDF resolution or reduce",
        "target voxel size to improve the actual SDF; increasing visualization sampling",
        "density alone does not add geometric information to the SDF.",
    ])
    readme = "\n".join(readme_lines) + "\n"
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("")
    print("SDF visualization export completed.")
    print(f"  ParaView entry: {multiblock_path}")
    print(f"  Metadata:      {metadata_path}")
    print(f"  Build time:    {build_seconds:.3f} s")
    print(f"  Sampling time: {sampling_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
