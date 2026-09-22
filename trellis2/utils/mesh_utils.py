from typing import Tuple, Dict
import numpy as np
from trimesh import grouping, util, remesh
import struct
import re
from plyfile import PlyData, PlyElement


def read_ply(filename):
    """
    Read a PLY file and return vertices, triangle faces, and quad faces.
    
    Args:
        filename (str): The file path to read from.
        
    Returns:
        vertices (np.ndarray): Array of shape [N, 3] containing vertex positions.
        tris (np.ndarray): Array of shape [M, 3] containing triangle face indices (empty if none).
        quads (np.ndarray): Array of shape [K, 4] containing quad face indices (empty if none).
    """
    with open(filename, 'rb') as f:
        # Read the header until 'end_header' is encountered
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header not found")
            header_bytes += line
            if b"end_header" in line:
                break
        header = header_bytes.decode('utf-8')
        
        # Determine if the file is in ASCII or binary format
        is_ascii = "ascii" in header
        
        # Extract the number of vertices and faces from the header using regex
        vertex_match = re.search(r'element vertex (\d+)', header)
        if vertex_match:
            num_vertices = int(vertex_match.group(1))
        else:
            raise ValueError("Vertex count not found in header")
            
        face_match = re.search(r'element face (\d+)', header)
        if face_match:
            num_faces = int(face_match.group(1))
        else:
            raise ValueError("Face count not found in header")
        
        vertices = []
        tris = []
        quads = []
        
        if is_ascii:
            # For ASCII format, read each line of vertex data (each line contains 3 floats)
            for _ in range(num_vertices):
                line = f.readline().decode('utf-8').strip()
                if not line: 
                    continue
                parts = line.split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            # Read face data, where the first number indicates the number of vertices for the face
            for _ in range(num_faces):
                line = f.readline().decode('utf-8').strip()
                if not line: 
                    continue
                parts = line.split()
                count = int(parts[0])
                indices = list(map(int, parts[1:]))
                if count == 3:
                    tris.append(indices)
                elif count == 4:
                    quads.append(indices)
                else:
                    # Skip faces with other numbers of vertices (can be extended as needed)
                    pass
        else:
            # For binary format: read directly from the binary stream
            # Each vertex consists of 3 floats (12 bytes per vertex)
            for _ in range(num_vertices):
                data = f.read(12)
                if len(data) < 12:
                    raise ValueError("Insufficient vertex data")
                v = struct.unpack('<fff', data)
                vertices.append(v)
            
            # Read face data from the binary stream
            for _ in range(num_faces):
                # First, read 1 byte indicating the number of vertices in the face
                count_data = f.read(1)
                if len(count_data) < 1:
                    raise ValueError("Failed to read face vertex count")
                count = struct.unpack('<B', count_data)[0]
                if count == 3:
                    data = f.read(12)  # 3 * 4 bytes
                    if len(data) < 12:
                        raise ValueError("Insufficient data for triangle face")
                    indices = struct.unpack('<3i', data)
                    tris.append(indices)
                elif count == 4:
                    data = f.read(16)  # 4 * 4 bytes
                    if len(data) < 16:
                        raise ValueError("Insufficient data for quad face")
                    indices = struct.unpack('<4i', data)
                    quads.append(indices)
                else:
                    # For faces with a different number of vertices, read count*4 bytes
                    data = f.read(count * 4)
                    # Skip or extend processing as needed
                    raise ValueError(f"Unsupported face with {count} vertices")
        
        # Convert lists to torch.Tensor
        vertices = np.array(vertices, dtype=np.float32)
        tris = np.array(tris, dtype=np.int32) if len(tris) > 0 else np.empty((0, 3), dtype=np.int32)
        quads = np.array(quads, dtype=np.int32) if len(quads) > 0 else np.empty((0, 4), dtype=np.int32)
        
        return vertices, tris, quads


def write_ply(
    filename: str,
    vertices: np.ndarray,
    tris: np.ndarray,
    quads: np.ndarray,
    vertex_colors: np.ndarray = None,
    ascii: bool = False
):
    """
    Write a mesh to a PLY file, with the option to save in ASCII or binary format,
    and optional per-vertex colors.
    
    Args:
        filename (str): The filename to write to.
        vertices (np.ndarray): [N, 3] The vertex positions.
        tris (np.ndarray): [M, 3] The triangle indices.
        quads (np.ndarray): [K, 4] The quad indices.
        vertex_colors (np.ndarray, optional): [N, 3] or [N, 4] UInt8 colors for each vertex (RGB or RGBA).
        ascii (bool): If True, write in ASCII format; otherwise binary little-endian.
    """
    import struct

    num_vertices = len(vertices)
    num_faces = len(tris) + len(quads)

    # Build header
    header_lines = [
        "ply",
        f"format {'ascii 1.0' if ascii else 'binary_little_endian 1.0'}",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]

    # Add vertex color properties if provided
    has_color = vertex_colors is not None
    if has_color:
        # Expect uint8 values 0-255
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        # Include alpha if RGBA
        if vertex_colors.shape[1] == 4:
            header_lines.append("property uchar alpha")

    header_lines += [
        f"element face {num_faces}",
        "property list uchar int vertex_index",
        "end_header",
        ""
    ]
    header = "\n".join(header_lines)

    mode = 'w' if ascii else 'wb'
    with open(filename, mode) as f:
        # Write header
        if ascii:
            f.write(header)
        else:
            f.write(header.encode('utf-8'))

        # Write vertex data
        for i, v in enumerate(vertices):
            if ascii:
                line = f"{v[0]} {v[1]} {v[2]}"
                if has_color:
                    col = vertex_colors[i]
                    line += ' ' + ' '.join(str(int(c)) for c in col)
                f.write(line + '\n')
            else:
                # pack xyz as floats
                f.write(struct.pack('<fff', *v))
                if has_color:
                    col = vertex_colors[i]
                    # pack as uchar
                    if col.shape[0] == 3:
                        f.write(struct.pack('<BBB', *col))
                    else:
                        f.write(struct.pack('<BBBB', *col))

        # Write face data
        if ascii:
            for tri in tris:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
            for quad in quads:
                f.write(f"4 {quad[0]} {quad[1]} {quad[2]} {quad[3]}\n")
        else:
            for tri in tris:
                f.write(struct.pack('<B3i', 3, *tri))
            for quad in quads:
                f.write(struct.pack('<B4i', 4, *quad))
                

def write_pbr_ply(
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    base_color: np.ndarray,
    metallic: np.ndarray,
    roughness: np.ndarray,
    alpha: np.ndarray,
    ascii: bool = False
):
    """
    Write a mesh to a PLY file, with the option to save in ASCII or binary format,
    and optional per-vertex colors.
    
    Args:
        filename (str): The filename to write to.
        vertices (np.ndarray): [N, 3] The vertex positions.
        faces (np.ndarray): [M, 3] The triangle indices.
        base_color (np.ndarray): [N, 3] UInt8 colors for each vertex (RGB).
        metallic (np.ndarray): [N] UInt8 values for metallicness.
        roughness (np.ndarray): [N] UInt8 values for roughness.
        alpha (np.ndarray): [N] UInt8 values for alpha.
        ascii (bool): If True, write in ASCII format; otherwise binary little-endian.
    """
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('metallic', 'u1'), ('roughness', 'u1'), ('alpha', 'u1')
    ]
    
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['red'] = base_color[:, 0]
    vertex_data['green'] = base_color[:, 1]
    vertex_data['blue'] = base_color[:, 2]
    vertex_data['metallic'] = metallic
    vertex_data['roughness'] = roughness
    vertex_data['alpha'] = alpha
    
    face_dtype = [
        ('vertex_indices', 'i4', (3,))
    ]
    
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces
    
    ply_data = PlyData([
        PlyElement.describe(vertex_data,'vertex'),
        PlyElement.describe(face_data, 'face'),
    ], text=ascii)
    ply_data.write(filename)


def snapshot_mesh_numpy(mesh):
    """Copy vertices/faces to CPU before a CUDA export path can kill the context."""
    vertices = mesh.vertices.detach().float().cpu().numpy().copy()
    faces = mesh.faces.detach().cpu().numpy().copy()
    return vertices, faces


def export_plain_glb(path, vertices_np, faces_np) -> str:
    """Write an untextured GLB. Safe after a CUDA context death."""
    import trimesh
    v = np.asarray(vertices_np, dtype=np.float32).copy()
    f = np.asarray(faces_np, dtype=np.int32)
    if v.shape[0] == 0 or f.shape[0] == 0:
        raise ValueError("Cannot export an empty mesh")
    # Same Y/Z swap as o_voxel.postprocess.to_glb
    y = v[:, 1].copy()
    v[:, 1] = v[:, 2]
    v[:, 2] = -y
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(path)
    return path


def export_textured_glb(
    mesh,
    path,
    *,
    decimation_target=None,
    texture_size=None,
    remesh=False,
) -> str:
    """
    Bake a textured GLB. Starts at the requested face count and texture size
    (defaults: 1,000,000 faces, 2048 texture). A Python OOM can retry a
    smaller bake. `device not ready` cannot — CUDA is already dead.
    """
    import o_voxel
    from trellis2.utils import offload
    from trellis2.utils.bake_limits import (
        RAISED_DECIMATION_TARGET,
        LOW_VRAM_TEXTURE_SIZE,
        bake_size_attempts,
        is_cuda_oom,
        is_cuda_context_dead,
    )

    if decimation_target is None:
        decimation_target = RAISED_DECIMATION_TARGET
    if texture_size is None:
        texture_size = LOW_VRAM_TEXTURE_SIZE

    attempts = bake_size_attempts(decimation_target, texture_size)
    for index, (faces, tex) in enumerate(attempts):
        print(
            f"Baking GLB: decimation_target={faces}, texture_size={tex}, remesh={remesh}"
        )
        offload.release_cuda_memory()
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=faces,
                texture_size=tex,
                remesh=remesh,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb.export(path, extension_webp=True)
            print(f"Wrote {path} (decimation_target={faces}, texture_size={tex})")
            return path
        except Exception as e:
            if is_cuda_context_dead(e):
                print(
                    f"Bake killed the CUDA context at decimation_target={faces}, "
                    f"texture_size={tex} ({e}).\n"
                    "This is not a Python OOM — Windows dropped the GPU. CUDA stays "
                    "dead until WSL is reset. In PowerShell run:\n"
                    "    wsl --shutdown\n"
                    "Then reopen Ubuntu. Default bake on 4 GB is 1,000,000 faces / "
                    "2048 texture. Do not pass TRELLIS_TEXTURE_SIZE=4096 on a 4 GB card."
                )
                raise
            if not is_cuda_oom(e) or index == len(attempts) - 1:
                raise
            print(
                f"Bake ran out of VRAM at decimation_target={faces}, "
                f"texture_size={tex} ({e}). Retrying a smaller bake."
            )
            offload.release_cuda_memory()
    raise RuntimeError("GLB bake failed")


def export_pbr_glb(mesh, path, *, texture_size=1024, decimation_target=100000, remesh=False) -> str:
    """
    Bake a PBR GLB via o_voxel. If cuMesh cleanup crashes (tiny 1536 meshes
    after integer-scaled occupancy), write a plain GLB from a CPU snapshot.
    """
    import o_voxel

    cpu_vertices, cpu_faces = snapshot_mesh_numpy(mesh)
    n_faces = int(cpu_faces.shape[0])
    try:
        if n_faces < 2048:
            raise RuntimeError(
                f"mesh too small for cuMesh cleanup ({n_faces} faces); using plain GLB"
            )
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(path, extension_webp=True)
        print(f"Wrote {path}")
        return path
    except Exception as e:
        print(f"[TRELLIS.2] PBR GLB export failed ({e})")
        export_plain_glb(path, cpu_vertices, cpu_faces)
        print(f"Wrote {path} (untextured fallback)")
        return path
