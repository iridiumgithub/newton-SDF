# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a texture SDF from an STL and export ParaView-ready diagnostics.

Quick start:
    uv sync --extra importers
    python scripts/visualize_sdf.py --input cases/hollow_cylinder.stl --output outputs/hollow_cylinder

Open outputs/hollow_cylinder/case.vtm in ParaView. The multiblock contains the
source mesh, Newton's extracted zero isosurface, a sampled 3-D SDF volume, and
three center-plane slices. The volume and slices contain the actual interpolated
Newton texture-SDF values, not values recomputed from the source STL.

STL files do not encode physical units. The default input unit is metres; pass
--input-unit mm (or cm/inch) when the STL coordinates use another unit.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf


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


_UNIT_TO_METERS = {
    "m": 1.0,
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "inch": 0.0254,
}


def _vec3_to_numpy(value: Any) -> np.ndarray:
    return np.asarray([float(value[i]) for i in range(3)], dtype=np.float64)


def _vtk_binary(values: np.ndarray, dtype: str) -> str:
    """Encode one VTK XML inline-binary DataArray with a UInt64 byte-count header."""
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    payload = struct.pack("<Q", array.nbytes) + array.tobytes(order="C")
    return base64.b64encode(payload).decode("ascii")


def _write_vtp(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    """Write triangle PolyData without requiring VTK or PyVista."""
    points = np.ascontiguousarray(points, dtype="<f4").reshape(-1, 3)
    faces = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)

    root = ET.Element(
        "VTKFile",
        {"type": "PolyData", "version": "1.0", "byte_order": "LittleEndian", "header_type": "UInt64"},
    )
    poly_data = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        poly_data,
        "Piece",
        {
            "NumberOfPoints": str(len(points)),
            "NumberOfVerts": "0",
            "NumberOfLines": "0",
            "NumberOfStrips": "0",
            "NumberOfPolys": str(len(faces)),
        },
    )

    ET.SubElement(piece, "PointData")
    ET.SubElement(piece, "CellData")
    points_element = ET.SubElement(piece, "Points")
    points_array = ET.SubElement(
        points_element,
        "DataArray",
        {"type": "Float32", "NumberOfComponents": "3", "format": "binary"},
    )
    points_array.text = _vtk_binary(points, "<f4")

    ET.SubElement(piece, "Verts")
    ET.SubElement(piece, "Lines")
    ET.SubElement(piece, "Strips")
    polys = ET.SubElement(piece, "Polys")
    connectivity = ET.SubElement(
        polys,
        "DataArray",
        {"type": "Int64", "Name": "connectivity", "format": "binary"},
    )
    connectivity.text = _vtk_binary(faces.reshape(-1), "<i8")
    offsets = ET.SubElement(
        polys,
        "DataArray",
        {"type": "Int64", "Name": "offsets", "format": "binary"},
    )
    offsets.text = _vtk_binary(np.arange(3, 3 * len(faces) + 1, 3, dtype=np.int64), "<i8")

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_vti(
    path: Path,
    dims: tuple[int, int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
    values: np.ndarray,
    array_name: str = "signed_distance_m",
) -> None:
    """Write a VTK ImageData file; x is the fastest-varying sample index."""
    nx, ny, nz = (int(v) for v in dims)
    expected_count = nx * ny * nz
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != expected_count:
        raise ValueError(f"VTI sample count mismatch: expected {expected_count}, got {len(values)}.")

    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    origin_text = " ".join(f"{float(v):.17g}" for v in origin)
    spacing_text = " ".join(f"{float(v):.17g}" for v in spacing)

    root = ET.Element(
        "VTKFile",
        {"type": "ImageData", "version": "1.0", "byte_order": "LittleEndian", "header_type": "UInt64"},
    )
    image_data = ET.SubElement(
        root,
        "ImageData",
        {"WholeExtent": extent, "Origin": origin_text, "Spacing": spacing_text},
    )
    piece = ET.SubElement(image_data, "Piece", {"Extent": extent})
    point_data = ET.SubElement(piece, "PointData", {"Scalars": array_name})
    data_array = ET.SubElement(
        point_data,
        "DataArray",
        {"type": "Float32", "Name": array_name, "NumberOfComponents": "1", "format": "binary"},
    )
    data_array.text = _vtk_binary(values, "<f4")
    ET.SubElement(piece, "CellData")

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_vtm(path: Path, datasets: list[tuple[str, Path]], output_dir: Path) -> None:
    """Write a ParaView multiblock index referencing the exported VTK datasets."""
    root = ET.Element(
        "VTKFile",
        {"type": "vtkMultiBlockDataSet", "version": "1.0", "byte_order": "LittleEndian"},
    )
    multiblock = ET.SubElement(root, "vtkMultiBlockDataSet")
    for index, (name, dataset_path) in enumerate(datasets):
        relative_path = dataset_path.relative_to(output_dir).as_posix()
        ET.SubElement(
            multiblock,
            "DataSet",
            {"index": str(index), "name": name, "file": relative_path},
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


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
) -> np.ndarray:
    """Sample the actual Newton texture SDF on a regular grid."""
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
    return output.numpy()


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
        "sampled_datasets": {},
    }

    if export_volume:
        dims = _fit_volume_dims(extents, volume_resolution, max_volume_points)
        spacing = extents / (np.asarray(dims, dtype=np.float64) - 1.0)
        values = _sample_sdf_grid(sdf_data_array, lower, spacing, dims, device)
        path = output_dir / "sdf_volume.vti"
        _write_vti(path, dims, lower, spacing, values)
        datasets.append(("Sampled SDF volume", path))
        sample_info["sampled_datasets"]["sdf_volume"] = {
            "file": path.name,
            "dimensions": list(dims),
            "spacing_m": spacing.tolist(),
            "sample_count": int(values.size),
            "minimum_m": float(np.min(values)),
            "maximum_m": float(np.max(values)),
            "negative_sample_fraction": float(np.mean(values < 0.0)),
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
            _write_vti(path, dims, origin, spacing, values)
            datasets.append((display_name, path))
            sample_info["sampled_datasets"][f"slice_{suffix}"] = {
                "file": path.relative_to(output_dir).as_posix(),
                "dimensions": list(dims),
                "origin_m": origin.tolist(),
                "spacing_m": spacing.tolist(),
                "sample_count": int(values.size),
                "minimum_m": float(np.min(values)),
                "maximum_m": float(np.max(values)),
                "negative_sample_fraction": float(np.mean(values < 0.0)),
            }
            print(f"  Exported {suffix.upper()} slice: {path} ({nu} x {nv})")

    return datasets, sample_info


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Newton texture SDF from an STL and export ParaView diagnostics."
    )
    parser.add_argument("--input", "--case", dest="input", type=Path, required=True, help="Input STL case file.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
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
    print(
        f"  Vertices: {mesh_info['vertex_count']}, triangles: {mesh_info['triangle_count']}, "
        f"watertight: {mesh_info['watertight']}"
    )
    if not mesh_info["watertight"] and args.sign_method == "auto":
        print("  WARNING: mesh is not watertight; auto mode will use winding-number signs. Inspect the exported slices.")
    elif not mesh_info["watertight"] and args.sign_method == "parity":
        print("  WARNING: mesh is not watertight; parity signs may be unreliable. Consider --sign-method normal.")

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
    build_seconds = time.perf_counter() - build_start
    print(f"  SDF build finished in {build_seconds:.3f} s")

    if sdf.is_empty() or sdf.to_texture_kernel_data() is None:
        raise RuntimeError("SDF construction produced an empty texture SDF.")

    datasets: list[tuple[str, Path]] = []
    source_path = output_dir / "source_mesh.vtp"
    _write_vtp(source_path, mesh.vertices, mesh.indices.reshape(-1, 3))
    datasets.append(("Source STL mesh", source_path))
    print(f"[3/5] Exported source mesh: {source_path}")

    print("[4/5] Extracting Newton zero isosurface ...")
    extraction_start = time.perf_counter()
    zero_mesh = sdf.extract_isomesh(isovalue=args.isovalue, device=device)
    extraction_seconds = time.perf_counter() - extraction_start
    zero_mesh_info: dict[str, Any]
    if zero_mesh is None or len(zero_mesh.indices) == 0:
        zero_mesh_info = {"extracted": False, "vertex_count": 0, "triangle_count": 0}
        print("  WARNING: Newton returned an empty isosurface.")
    else:
        zero_path = output_dir / "newton_zero_isosurface.vtp"
        zero_faces = np.asarray(zero_mesh.indices, dtype=np.int64).reshape(-1, 3)
        _write_vtp(zero_path, zero_mesh.vertices, zero_faces)
        datasets.append(("Newton extracted zero isosurface", zero_path))
        zero_mesh_info = {
            "extracted": True,
            "vertex_count": int(len(zero_mesh.vertices)),
            "triangle_count": int(len(zero_faces)),
            "isovalue_m": float(args.isovalue),
        }
        print(
            f"  Exported zero surface: {zero_path} "
            f"({zero_mesh_info['vertex_count']} vertices, {zero_mesh_info['triangle_count']} triangles)"
        )

    print("[5/5] Sampling the actual interpolated texture SDF ...")
    sampling_start = time.perf_counter()
    sample_datasets, sample_info = _export_sdf_diagnostics(
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
    datasets.extend(sample_datasets)
    sampling_seconds = time.perf_counter() - sampling_start

    multiblock_path = output_dir / "case.vtm"
    _write_vtm(multiblock_path, datasets, output_dir)

    texture_data = sdf.to_texture_kernel_data()
    coarse_texture = sdf._coarse_texture
    subgrid_texture = sdf._subgrid_texture
    coarse_texture_dimensions = (
        [int(coarse_texture.width), int(coarse_texture.height), int(coarse_texture.depth)]
        if coarse_texture is not None
        else None
    )
    coarse_cell_dimensions = (
        [dimension - 1 for dimension in coarse_texture_dimensions]
        if coarse_texture_dimensions is not None
        else None
    )
    subgrid_size = int(texture_data.subgrid_size)
    fine_grid_cell_dimensions = (
        [dimension * subgrid_size for dimension in coarse_cell_dimensions]
        if coarse_cell_dimensions is not None
        else None
    )
    sample_info["texture_layout"] = {
        "coarse_texture_dimensions": coarse_texture_dimensions,
        "coarse_cell_dimensions": coarse_cell_dimensions,
        "fine_grid_cell_dimensions": fine_grid_cell_dimensions,
        "fine_grid_sample_dimensions": (
            [dimension + 1 for dimension in fine_grid_cell_dimensions]
            if fine_grid_cell_dimensions is not None
            else None
        ),
        "subgrid_texture_dimensions": (
            [int(subgrid_texture.width), int(subgrid_texture.height), int(subgrid_texture.depth)]
            if subgrid_texture is not None
            else None
        ),
        "subgrid_size": subgrid_size,
        "num_subgrids": int(texture_data.num_subgrids),
        "paired_samples": bool(texture_data.paired_samples),
    }

    metadata = {
        "tool": "Newton SDF Inspector",
        "input": mesh_info,
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
            "texture_format": args.texture_format,
            "paired_samples": not args.no_paired_samples,
            "build_seconds": build_seconds,
            "isovalue_m": args.isovalue,
        },
        "extraction": {
            **zero_mesh_info,
            "seconds": extraction_seconds,
        },
        "sampling": {
            **sample_info,
            "seconds": sampling_seconds,
        },
        "outputs": [path.relative_to(output_dir).as_posix() for _, path in datasets] + ["case.vtm"],
        "notes": [
            "The VTI values are sampled from Newton's interpolated texture SDF.",
            "The Newton zero isosurface is extracted by Newton's texture-SDF Marching Cubes path.",
            "STL has no physical-unit metadata; input coordinates were scaled according to input_unit.",
            "This diagnostic currently visualizes the existing single-field Newton SDF path; it does not yet visualize the experimental dual-field sampler.",
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_lines = [
        "Newton SDF Inspector output",
        "",
        "1. Open case.vtm in ParaView to load all exported datasets together.",
        "2. Toggle Source STL mesh and Newton extracted zero isosurface to compare geometry.",
    ]
    if not args.no_volume:
        readme_lines.extend([
            "3. Select sdf_volume.vti and apply the Contour filter with value 0.0 to extract",
            "   the zero level set from the sampled interpolated SDF.",
        ])
    if not args.no_slices:
        readme_lines.extend([
            "4. Open the three files under slices/ to inspect signed-distance cross sections.",
            "   Color by signed_distance_m; the material side should be negative for a correctly",
            "   oriented closed solid.",
        ])
    readme_lines.extend([
        "5. Check metadata.json for mesh watertightness, SDF parameters, texture layout,",
        "   sample dimensions, and value ranges.",
        "",
        "The full 3-D volume is capped by --max-volume-points. Center slices use",
        "--slice-resolution and --max-slice-points. Increase SDF resolution or reduce",
        "target voxel size to improve the actual SDF; increasing visualization sampling",
        "density alone does not add geometric information to the SDF.",
    ])
    readme = "\\n".join(readme_lines) + "\\n"
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
