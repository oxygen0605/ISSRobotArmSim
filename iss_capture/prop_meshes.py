"""Batch static cabin props by material to keep Gazebo camera rendering light.

Only the two decorative cabin models are compiled. Robot, sponge, droplet,
collisions, cameras, and the original work surface are not touched.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .kinematics import rotation


def primitive_faces(geometry):
    """Yield triangles as (vertices, smooth/flat vertex normals)."""
    def quad(v, n):
        for ids in ([0, 1, 2], [0, 2, 3]):
            points = np.asarray(v)[ids]
            if np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) > 1e-14:
                yield points, np.asarray(n)[ids]
    if geometry.find("box") is not None:
        x, y, z = np.fromstring(geometry.findtext("box/size"), sep=" ") / 2
        faces = [([(x,-y,-z),(x,y,-z),(x,y,z),(x,-y,z)], (1,0,0)),
                 ([(-x,y,-z),(-x,-y,-z),(-x,-y,z),(-x,y,z)], (-1,0,0)),
                 ([(-x,y,-z),(-x,y,z),(x,y,z),(x,y,-z)], (0,1,0)),
                 ([(-x,-y,z),(-x,-y,-z),(x,-y,-z),(x,-y,z)], (0,-1,0)),
                 ([(-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)], (0,0,1)),
                 ([(-x,y,-z),(x,y,-z),(x,-y,-z),(-x,-y,-z)], (0,0,-1))]
        for vertices, normal in faces:
            yield from quad(vertices, [normal] * 4)
    elif geometry.find("cylinder") is not None:
        r, h = float(geometry.findtext("cylinder/radius")), float(geometry.findtext("cylinder/length")) / 2
        segments = 48 if r >= .08 else (8 if r < .008 else 16)
        for i in range(segments):
            a, b = i * 2 * np.pi / segments, (i + 1) * 2 * np.pi / segments
            na, nb = np.array([np.cos(a),np.sin(a),0]), np.array([np.cos(b),np.sin(b),0])
            va, vb = na * r, nb * r
            yield from quad([va+[0,0,-h],vb+[0,0,-h],vb+[0,0,h],va+[0,0,h]], [na,nb,nb,na])
            yield np.array([[0,0,h],va+[0,0,h],vb+[0,0,h]]), np.array([[0,0,1]] * 3)
            yield np.array([[0,0,-h],vb+[0,0,-h],va+[0,0,-h]]), np.array([[0,0,-1]] * 3)
    elif geometry.find("sphere") is not None:
        r = float(geometry.findtext("sphere/radius"))
        def point(theta, phi):
            return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        for j in range(12):
            for i in range(24):
                a,b = j*np.pi/12,(j+1)*np.pi/12
                c,d = i*2*np.pi/24,(i+1)*2*np.pi/24
                normals = np.array([point(a,c),point(b,c),point(b,d),point(a,d)])
                yield from quad(normals * r, normals)
    else:
        raise ValueError("Unsupported decorative primitive")


def compile_cabin_meshes(world, directory):
    directory = Path(directory).resolve()
    groups_count, originals, retained = 0, 0, 0
    for name in ("iss_cabin", "iss_experiment_fixture"):
        model = world.find(f"model[@name='{name}']")
        if model is None:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        link = model.find("link")
        groups = {}
        for visual in list(link.findall("visual")):
            # Authored decal meshes retain their UVs/PBR textures. Primitive
            # batching deliberately does not flatten textured surfaces.
            if visual.find('geometry/mesh') is not None:
                retained += 1
                originals += 1
                groups_count += 1
                continue
            key = (ET.tostring(visual.find("material")), visual.findtext("transparency", "0"))
            groups.setdefault(key, []).append(visual)
        for index, ((material, transparency), visuals) in enumerate(groups.items()):
            path = directory / f"{name}_{index}.obj"
            lines, offset = ["# Procedural cabin geometry; metres; normals supplied."], 1
            for visual in visuals:
                pose = np.fromstring(visual.findtext("pose", "0 0 0 0 0 0"), sep=" ")
                r,p,y = pose[3:]
                matrix = rotation([0,0,1],y) @ rotation([0,1,0],p) @ rotation([1,0,0],r)
                for points, normals in primitive_faces(visual.find("geometry")):
                    points, normals = points @ matrix.T + pose[:3], normals @ matrix.T
                    lines.extend("v " + " ".join(f"{v:.8g}" for v in point) for point in points)
                    lines.extend("vn " + " ".join(f"{v:.8g}" for v in normal) for normal in normals)
                    lines.append("f " + " ".join(f"{i}//{i}" for i in range(offset, offset + 3)))
                    offset += 3
                link.remove(visual)
            path.write_text("\n".join(lines) + "\n")
            merged = ET.SubElement(link, "visual", name=f"material_group_{index}")
            ET.SubElement(merged, "cast_shadows").text = "false"
            ET.SubElement(merged, "transparency").text = transparency
            merged.append(ET.fromstring(material))
            mesh = ET.SubElement(ET.SubElement(merged, "geometry"), "mesh")
            ET.SubElement(mesh, "uri").text = path.as_uri()
            originals += len(visuals)
        groups_count += len(groups)
    return {"original_visuals": originals, "batched_visuals": groups_count,
            "retained_textured_meshes": retained}
