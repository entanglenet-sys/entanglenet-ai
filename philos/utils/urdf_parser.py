"""
URDF Parser for PHILOS digital twin.

Parses the UR5e URDF file and provides:
  - Kinematic chain with proper DH-like transforms
  - Forward kinematics using homogeneous transformation matrices
  - Joint limits, masses, inertias for physics simulation
  - A JSON-serializable description for the Three.js 3D twin

This is the SINGLE SOURCE OF TRUTH — both the Python simulation
and the browser-side 3D visualization derive from this parser.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────


@dataclass
class URDFGeometry:
    """Parsed visual/collision geometry."""
    shape: str                       # "cylinder", "box", "sphere"
    radius: float = 0.0
    length: float = 0.0
    size: Tuple[float, ...] = ()     # for box
    origin_xyz: Tuple[float, float, float] = (0, 0, 0)
    origin_rpy: Tuple[float, float, float] = (0, 0, 0)
    color_rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    material_name: str = ""


@dataclass
class URDFInertial:
    """Parsed inertial properties."""
    mass: float = 0.0
    origin_xyz: Tuple[float, float, float] = (0, 0, 0)
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0


@dataclass
class URDFLink:
    """Parsed link."""
    name: str
    visual: Optional[URDFGeometry] = None
    collision: Optional[URDFGeometry] = None
    inertial: Optional[URDFInertial] = None


@dataclass
class URDFJoint:
    """Parsed joint."""
    name: str
    type: str                         # "revolute", "fixed"
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float] = (0, 0, 0)
    origin_rpy: Tuple[float, float, float] = (0, 0, 0)
    axis: Tuple[float, float, float] = (0, 0, 1)
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    damping: float = 0.0
    friction: float = 0.0


@dataclass
class URDFRobot:
    """Full parsed URDF robot model."""
    name: str
    links: Dict[str, URDFLink] = field(default_factory=dict)
    joints: Dict[str, URDFJoint] = field(default_factory=dict)
    # Ordered chain from base to tip
    joint_chain: List[str] = field(default_factory=list)
    # Only the actuated (revolute) joints in order
    actuated_joints: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────


def _parse_vec(text: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in text.strip().split())


def _parse_origin(elem) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    origin = elem.find("origin")
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin is not None:
        if "xyz" in origin.attrib:
            xyz = tuple(float(x) for x in origin.attrib["xyz"].split())
        if "rpy" in origin.attrib:
            rpy = tuple(float(x) for x in origin.attrib["rpy"].split())
    return xyz, rpy


def _parse_geometry(vis_elem) -> Optional[URDFGeometry]:
    if vis_elem is None:
        return None
    geom_elem = vis_elem.find("geometry")
    if geom_elem is None:
        return None

    origin_xyz, origin_rpy = _parse_origin(vis_elem)

    # Material color
    color_rgba = (0.5, 0.5, 0.5, 1.0)
    mat_name = ""
    mat = vis_elem.find("material")
    if mat is not None:
        mat_name = mat.attrib.get("name", "")
        col = mat.find("color")
        if col is not None and "rgba" in col.attrib:
            color_rgba = tuple(float(x) for x in col.attrib["rgba"].split())

    cyl = geom_elem.find("cylinder")
    if cyl is not None:
        return URDFGeometry(
            shape="cylinder",
            radius=float(cyl.attrib["radius"]),
            length=float(cyl.attrib["length"]),
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            color_rgba=color_rgba,
            material_name=mat_name,
        )
    box = geom_elem.find("box")
    if box is not None:
        return URDFGeometry(
            shape="box",
            size=tuple(float(x) for x in box.attrib["size"].split()),
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            color_rgba=color_rgba,
            material_name=mat_name,
        )
    sphere = geom_elem.find("sphere")
    if sphere is not None:
        return URDFGeometry(
            shape="sphere",
            radius=float(sphere.attrib["radius"]),
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            color_rgba=color_rgba,
            material_name=mat_name,
        )
    return None


def _parse_inertial(link_elem) -> Optional[URDFInertial]:
    iner = link_elem.find("inertial")
    if iner is None:
        return None
    mass_el = iner.find("mass")
    mass = float(mass_el.attrib["value"]) if mass_el is not None else 0.0
    origin_xyz, _ = _parse_origin(iner)
    inertia_el = iner.find("inertia")
    ixx = iyy = izz = 0.0
    if inertia_el is not None:
        ixx = float(inertia_el.attrib.get("ixx", 0))
        iyy = float(inertia_el.attrib.get("iyy", 0))
        izz = float(inertia_el.attrib.get("izz", 0))
    return URDFInertial(mass=mass, origin_xyz=origin_xyz, ixx=ixx, iyy=iyy, izz=izz)


# ──────────────────────────────────────────────────────────────────
# Main parser
# ──────────────────────────────────────────────────────────────────

# Cache for material colors defined once with <color> and reused by name
_material_colors: Dict[str, Tuple[float, float, float, float]] = {}


def parse_urdf(path: str | Path) -> URDFRobot:
    """Parse a URDF file and return a structured robot model."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    robot = URDFRobot(name=root.attrib.get("name", "robot"))

    # First pass: collect all material definitions with colors
    for link_elem in root.findall("link"):
        for vis in [link_elem.find("visual")]:
            if vis is None:
                continue
            mat = vis.find("material")
            if mat is not None:
                name = mat.attrib.get("name", "")
                col = mat.find("color")
                if col is not None and "rgba" in col.attrib and name:
                    _material_colors[name] = tuple(
                        float(x) for x in col.attrib["rgba"].split()
                    )

    # Parse links
    for link_elem in root.findall("link"):
        name = link_elem.attrib["name"]
        vis_geom = _parse_geometry(link_elem.find("visual"))
        # Resolve material name → color if color was defined elsewhere
        if vis_geom and not vis_geom.color_rgba == (0.5, 0.5, 0.5, 1.0):
            pass  # already has color
        elif vis_geom and vis_geom.material_name in _material_colors:
            vis_geom.color_rgba = _material_colors[vis_geom.material_name]

        robot.links[name] = URDFLink(
            name=name,
            visual=vis_geom,
            collision=_parse_geometry(link_elem.find("collision")),
            inertial=_parse_inertial(link_elem),
        )

    # Parse joints
    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        jtype = joint_elem.attrib["type"]
        parent = joint_elem.find("parent").attrib["link"]
        child = joint_elem.find("child").attrib["link"]
        origin_xyz, origin_rpy = _parse_origin(joint_elem)

        axis = (0.0, 0.0, 1.0)
        axis_el = joint_elem.find("axis")
        if axis_el is not None and "xyz" in axis_el.attrib:
            axis = tuple(float(x) for x in axis_el.attrib["xyz"].split())

        lower = upper = velocity = effort = damping = friction = 0.0
        limit_el = joint_elem.find("limit")
        if limit_el is not None:
            lower = float(limit_el.attrib.get("lower", 0))
            upper = float(limit_el.attrib.get("upper", 0))
            velocity = float(limit_el.attrib.get("velocity", 0))
            effort = float(limit_el.attrib.get("effort", 0))

        dyn_el = joint_elem.find("dynamics")
        if dyn_el is not None:
            damping = float(dyn_el.attrib.get("damping", 0))
            friction = float(dyn_el.attrib.get("friction", 0))

        robot.joints[name] = URDFJoint(
            name=name,
            type=jtype,
            parent=parent,
            child=child,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
            lower=lower,
            upper=upper,
            velocity=velocity,
            effort=effort,
            damping=damping,
            friction=friction,
        )

    # Build ordered joint chain (walk from world root)
    _build_chain(robot)
    return robot


def _build_chain(robot: URDFRobot):
    """Walk the tree from root to tips, recording joint order."""
    # Find root link (one that is never a child)
    children = {j.child for j in robot.joints.values()}
    parents = {j.parent for j in robot.joints.values()}
    roots = parents - children
    if not roots:
        return
    root = sorted(roots)[0]  # deterministic

    visited = set()
    queue = [root]
    while queue:
        link_name = queue.pop(0)
        if link_name in visited:
            continue
        visited.add(link_name)
        for jname, j in robot.joints.items():
            if j.parent == link_name and jname not in robot.joint_chain:
                robot.joint_chain.append(jname)
                if j.type == "revolute":
                    robot.actuated_joints.append(jname)
                queue.append(j.child)


# ──────────────────────────────────────────────────────────────────
# Forward Kinematics (proper 3D with 4×4 homogeneous transforms)
# ──────────────────────────────────────────────────────────────────


def _rpy_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Roll-Pitch-Yaw (XYZ intrinsic) to 3×3 rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])


def _axis_rotation(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    """Rotation matrix around an arbitrary unit axis by angle (Rodrigues)."""
    ax = np.array(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    K = np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _make_transform(xyz, rpy, axis=None, angle=0.0) -> np.ndarray:
    """Build 4×4 homogeneous transform: T = Trans(xyz) · Rot(rpy) · Rot(axis, angle)."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    if axis is not None and abs(angle) > 1e-12:
        R_joint = np.eye(4)
        R_joint[:3, :3] = _axis_rotation(axis, angle)
        T = T @ R_joint
    return T


def forward_kinematics(
    robot: URDFRobot,
    joint_angles: Dict[str, float] | np.ndarray | list,
) -> Dict[str, np.ndarray]:
    """
    Compute 4×4 world-frame transforms for every link in the chain.

    Parameters:
        robot: Parsed URDF model
        joint_angles: Either a dict {joint_name: angle} or an array
                      matching the order of robot.actuated_joints

    Returns:
        Dict mapping link_name → 4×4 numpy transform
    """
    # Normalise joint_angles to dict
    if not isinstance(joint_angles, dict):
        angles_list = list(joint_angles)
        angle_dict = {}
        for i, jname in enumerate(robot.actuated_joints):
            angle_dict[jname] = angles_list[i] if i < len(angles_list) else 0.0
    else:
        angle_dict = joint_angles

    # Walk the chain, accumulating transforms
    link_transforms: Dict[str, np.ndarray] = {}
    # Root link at origin
    children_set = {j.child for j in robot.joints.values()}
    parents_set = {j.parent for j in robot.joints.values()}
    root_links = parents_set - children_set
    for rl in root_links:
        link_transforms[rl] = np.eye(4)

    for jname in robot.joint_chain:
        j = robot.joints[jname]
        parent_T = link_transforms.get(j.parent, np.eye(4))

        angle = angle_dict.get(jname, 0.0)
        if j.type == "revolute":
            # Clamp to joint limits
            if j.lower != j.upper:
                angle = max(j.lower, min(j.upper, angle))
            joint_T = _make_transform(j.origin_xyz, j.origin_rpy, j.axis, angle)
        else:
            joint_T = _make_transform(j.origin_xyz, j.origin_rpy)

        link_transforms[j.child] = parent_T @ joint_T

    return link_transforms


def get_end_effector_pose(
    robot: URDFRobot,
    joint_angles: Dict[str, float] | np.ndarray | list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (position_3, rotation_3x3) of the tool0 frame.
    """
    transforms = forward_kinematics(robot, joint_angles)
    tool_T = transforms.get("tool0", np.eye(4))
    return tool_T[:3, 3].copy(), tool_T[:3, :3].copy()


# ──────────────────────────────────────────────────────────────────
# JSON export for dashboard 3D twin
# ──────────────────────────────────────────────────────────────────


def urdf_to_json(robot: URDFRobot) -> dict:
    """
    Export the URDF model as a JSON-serializable dict that the
    dashboard Three.js code can use to build the 3D scene.

    The structure mirrors the URDF hierarchy so the browser can
    construct the same kinematic tree.
    """
    links_json = {}
    for name, link in robot.links.items():
        lj = {"name": name}
        if link.visual:
            v = link.visual
            lj["visual"] = {
                "shape": v.shape,
                "radius": v.radius,
                "length": v.length,
                "size": list(v.size),
                "origin_xyz": list(v.origin_xyz),
                "origin_rpy": list(v.origin_rpy),
                "color_rgba": list(v.color_rgba),
            }
        if link.inertial:
            lj["mass"] = link.inertial.mass
        links_json[name] = lj

    joints_json = {}
    for name, joint in robot.joints.items():
        joints_json[name] = {
            "name": name,
            "type": joint.type,
            "parent": joint.parent,
            "child": joint.child,
            "origin_xyz": list(joint.origin_xyz),
            "origin_rpy": list(joint.origin_rpy),
            "axis": list(joint.axis),
            "lower": joint.lower,
            "upper": joint.upper,
            "velocity": joint.velocity,
            "effort": joint.effort,
        }

    return {
        "name": robot.name,
        "links": links_json,
        "joints": joints_json,
        "joint_chain": robot.joint_chain,
        "actuated_joints": robot.actuated_joints,
        "num_actuated": len(robot.actuated_joints),
    }


# ──────────────────────────────────────────────────────────────────
# Convenience: default URDF path
# ──────────────────────────────────────────────────────────────────

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "simulation" / "assets"
DEFAULT_URDF_PATH = _ASSETS_DIR / "ur5e.urdf"


def load_default_robot() -> URDFRobot:
    """Load the project's default UR5e URDF."""
    return parse_urdf(DEFAULT_URDF_PATH)


# ──────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    robot = load_default_robot()
    print(f"Robot: {robot.name}")
    print(f"Links: {list(robot.links.keys())}")
    print(f"Joints: {list(robot.joints.keys())}")
    print(f"Actuated: {robot.actuated_joints}")
    print(f"Joint chain: {robot.joint_chain}")

    # Zero-position FK
    zeros = [0.0] * len(robot.actuated_joints)
    transforms = forward_kinematics(robot, zeros)
    pos, rot = get_end_effector_pose(robot, zeros)
    print(f"\nZero-config tool0 position: {pos}")
    print(f"Zero-config tool0 Z-axis: {rot[:, 2]}")

    # JSON export size
    jdata = urdf_to_json(robot)
    print(f"\nJSON export: {len(json.dumps(jdata))} bytes")
    print(f"Num actuated joints: {jdata['num_actuated']}")
