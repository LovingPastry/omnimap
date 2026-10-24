"""Deployment must preserve existing scripts and refuse package conflicts."""
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT/"scripts/deploy_sumi_kinematics.py"


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.package = Path(temporary.name)/"src/robot_else_nodes"
        (self.package/"scripts").mkdir(parents=True)
        (self.package/"package.xml").write_text('<package format="2"></package>')
        self.original = {"camera_twist_bridge.py": "# original plain bridge\n",
                         "camera_twist_bridge_kf.py": "# original KF bridge\n"}
        for name, text in self.original.items():
            (self.package/"scripts"/name).write_text(text)

    def run_deploy(self):
        return subprocess.run([sys.executable, str(DEPLOY), "--robot-package", str(self.package)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)

    def test_backup_entrypoints_and_idempotence(self):
        first = self.run_deploy()
        self.assertEqual(first.returncode, 0, first.stderr)
        manifest = self.package/"package.xml"
        self.assertEqual({node.text for node in ET.fromstring(manifest.read_text()).findall("exec_depend")},
                         {"omnimap_msgs", "omnimap_kinematics"})
        self.assertFalse(manifest.stat().st_mode & 0o111)
        manifest_before = (manifest.read_bytes(), manifest.stat().st_mtime_ns)
        for name, original in self.original.items():
            target = self.package/"scripts"/name
            backups = list(target.parent.glob(name+".before_omnimap_*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            self.assertTrue(target.stat().st_mode & 0o111)
            compile(target.read_text(), str(target), "exec")
            self.assertIn("_filter_mode:=kalman" if "_kf" in name else "_filter_mode:=none", target.read_text())
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.package/"scripts").iterdir()}
        second = self.run_deploy()
        self.assertEqual(second.returncode, 0, second.stderr)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.package/"scripts").iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(manifest_before, (manifest.read_bytes(), manifest.stat().st_mtime_ns))
        for name in ("omnimap_msgs", "omnimap_kinematics"):
            self.assertEqual((self.package.parent/name).resolve(), ROOT/"ros_ws/src"/name)

    def test_package_conflict_is_rejected_before_any_script_edit(self):
        (self.package.parent/"omnimap_kinematics").mkdir()
        result = self.run_deploy()
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing to replace another package", result.stderr)
        self.assertFalse((self.package.parent/"omnimap_msgs").exists())
        for name, original in self.original.items():
            self.assertEqual((self.package/"scripts"/name).read_text(), original)

    def test_missing_legacy_script_is_rejected_before_creating_links(self):
        (self.package/"scripts/camera_twist_bridge_kf.py").unlink()
        result = self.run_deploy()
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.package.parent/"omnimap_msgs").exists())
        self.assertEqual((self.package/"scripts/camera_twist_bridge.py").read_text(),
                         self.original["camera_twist_bridge.py"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
