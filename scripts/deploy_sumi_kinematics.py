#!/usr/bin/env python3
"""Install version-controlled bridges into the existing sumi robot package.

Only source files/symlinks are changed. No ROS master, driver or robot is started.
Existing scripts are backed up before replacement; repeat runs are idempotent.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-package", type=Path,
                        default=Path.home()/"ros/catkin_ws/src/robot_else_nodes")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    target = args.robot_package.resolve()
    source = repo/"ros_ws/src/omnimap_kinematics/scripts/camera_twist_bridge.py"
    messages = repo/"ros_ws/devel/lib/python3/dist-packages"
    if not (target/"package.xml").is_file() or not (target/"scripts").is_dir():
        parser.error("robot package not found: " + str(target))
    if (any(not (messages/name).is_file() for name in (
            "omnimap_msgs/msg/_BridgeStatus.py", "omnimap_msgs/msg/_KinematicModel.py",
            "omnimap_msgs/srv/_EvaluateKinematicModel.py")) or
            not (repo/"ros_ws/devel/lib/omnimap_kinematics/omnimap_kinematic_monitor").is_file()):
        parser.error("build first: ./scripts/build_ros_ws.sh --with-kinematics")
    # Preflight all conflicts before changing either package or bridge.
    links = [(target.parent/name, repo/"ros_ws/src"/name)
             for name in ("omnimap_msgs", "omnimap_kinematics")]
    for link, destination in links:
        if (link.exists() or link.is_symlink()) and link.resolve() != destination.resolve():
            parser.error("refusing to replace another package: " + str(link))
    replacements = []
    manifest = target/"package.xml"
    manifest_text = manifest.read_text()
    try:
        root = ET.fromstring(manifest_text)
    except ET.ParseError as exc:
        parser.error("invalid package.xml: " + str(exc))
    dependency_tag = "exec_depend" if int(root.get("format", "1")) >= 2 else "run_depend"
    present = {node.text.strip() for tag in ("depend", dependency_tag)
               for node in root.findall(tag) if node.text}
    missing = [name for name in ("omnimap_msgs", "omnimap_kinematics") if name not in present]
    if missing:
        if "</package>" not in manifest_text:
            parser.error("package.xml must have a closing package element")
        additions = "".join(f"  <{dependency_tag}>{name}</{dependency_tag}>\n" for name in missing)
        replacements.append((manifest, manifest_text.replace("</package>", additions+"</package>")))
    for name, mode in (("camera_twist_bridge.py", "none"), ("camera_twist_bridge_kf.py", "kalman")):
        path = target/"scripts"/name
        if not path.is_file() or path.is_symlink():
            parser.error("expected an existing regular bridge script: " + str(path))
        content = ("#!/usr/bin/env python3\n"
                   '"""OmniMap managed entrypoint; see scripts/deploy_sumi_kinematics.py."""\n'
                   "import runpy\nimport sys\n"
                   f"sys.path.insert(0, {str(messages)!r})\n"
                   f"if not any(arg.startswith('_filter_mode:=') for arg in sys.argv):\n"
                   f"    sys.argv.append('_filter_mode:={mode}')\n"
                   f"runpy.run_path({str(source)!r}, run_name='__main__')\n")
        if path.read_text() != content:
            replacements.append((path, content))
    for link, destination in links:
        if not link.is_symlink() and not link.exists():
            link.symlink_to(destination, target_is_directory=True)
            print(f"linked {link} -> {destination}")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for path, content in replacements:
        backup = path.with_name(path.name+".before_omnimap_"+suffix)
        shutil.copy2(path, backup)
        path.write_text(content)
        if path.suffix == ".py":
            path.chmod(path.stat().st_mode | 0o111)
        print(f"updated {path}; backup {backup}")
    print("Deployment ready. No ROS nodes were started.")
    print(f"source {repo}/ros_ws/devel/setup.bash --extend")
    print("roslaunch omnimap_kinematics diagnostics.launch start_bridge:=true")


if __name__ == "__main__":
    main()
