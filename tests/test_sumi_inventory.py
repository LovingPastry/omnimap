"""An effective Servo namespace must determine the collected runtime topics."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from xmlrpc.server import SimpleXMLRPCServer

PATH = Path(__file__).resolve().parents[1]/"scripts/collect_sumi_kinematics.py"
SPEC = importlib.util.spec_from_file_location("sumi_inventory", PATH)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def test_collects_effective_topics_from_parameter_server(self):
        config = {"cartesian_command_in_topic": "cartesian", "status_topic": "feedback",
                  "joint_topic": "/robot/measured_joints"}
        with SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False) as master:
            def get_param(caller, name):
                self.assertEqual(name, "/custom/servo")
                return [1, "ok", config]
            master.register_function(get_param, "getParam")
            worker = threading.Thread(target=master.serve_forever, daemon=True)
            worker.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)/"inventory.txt"
                    commands = []
                    def command(args, **kwargs):
                        commands.append(args)
                        return subprocess.CompletedProcess(args, 0, "fixture\n")
                    uri = "http://127.0.0.1:%d" % master.server_address[1]
                    with patch.dict(os.environ, ROS_MASTER_URI=uri), \
                            patch.object(sys, "argv", [str(PATH), "--servo-ns", "/custom/servo",
                                "--input-topic", "/camera/command", "--output", str(output)]), \
                            patch.object(subprocess, "run", side_effect=command):
                        self.assertEqual(inventory.main(), 0)
                    topics = {args[-1] for args in commands if args[:2] == ["rostopic", "info"]}
                    self.assertTrue({"/custom/servo/cartesian", "/custom/servo/feedback",
                                     "/robot/measured_joints", "/camera/command"}.issubset(topics))
                    self.assertNotIn("/servo_server/status", topics)
                    self.assertIn('"status_topic": "feedback"', output.read_text())
                    master.register_function(lambda caller, name: [-1, "missing", 0], "getParam")
                    output_missing = Path(directory)/"missing_config.txt"
                    commands.clear()
                    with patch.dict(os.environ, ROS_MASTER_URI=uri), \
                            patch.object(sys, "argv", [str(PATH), "--servo-ns", "/custom/servo",
                                "--output", str(output_missing)]), \
                            patch.object(subprocess, "run", side_effect=command):
                        self.assertEqual(inventory.main(), 2)
                    self.assertIn("CONFIG_UNVERIFIED", output_missing.read_text())
                    self.assertNotIn(["rostopic", "info", "/servo_server/status"], commands)
            finally:
                master.shutdown()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
