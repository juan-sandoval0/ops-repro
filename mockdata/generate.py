#!/usr/bin/env python3
"""
Mock data generator for robot run logs.
Generates realistic JSONL logs with timestamps, observations, actions, and errors.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import argparse


class RobotLogGenerator:
    """Generates mock robot run logs with various failure modes."""

    # Action-dependent error types with weights
    # Maps action types to possible errors and their probabilities
    ACTION_ERROR_MAP = {
        "grasp_object": {
            "gripper_slip": 0.25,
            "grasp_failure": 0.20,
            "object_not_found": 0.15,
            "force_threshold_exceeded": 0.15,
            "vision_failure": 0.10,
            "vacuum_loss": 0.10,
            "timeout": 0.05,
        },
        "release_object": {
            "gripper_slip": 0.30,
            "position_drift": 0.20,
            "execution_error": 0.20,
            "force_threshold_exceeded": 0.15,
            "timeout": 0.10,
            "sensor_malfunction": 0.05,
        },
        "open_gripper": {
            "motor_overheating": 0.25,
            "force_threshold_exceeded": 0.20,
            "calibration_error": 0.20,
            "sensor_malfunction": 0.15,
            "timeout": 0.10,
            "emergency_stop": 0.10,
        },
        "close_gripper": {
            "motor_overheating": 0.25,
            "force_threshold_exceeded": 0.20,
            "gripper_slip": 0.15,
            "vacuum_loss": 0.15,
            "timeout": 0.10,
            "calibration_error": 0.10,
            "sensor_malfunction": 0.05,
        },
        "move_to_position": {
            "collision_detected": 0.25,
            "workspace_violation": 0.20,
            "joint_limit_exceeded": 0.15,
            "path_deviation": 0.15,
            "position_drift": 0.10,
            "singularity_detected": 0.10,
            "timeout": 0.05,
        },
        "execute_trajectory": {
            "collision_detected": 0.20,
            "execution_error": 0.20,
            "path_deviation": 0.15,
            "joint_limit_exceeded": 0.15,
            "motor_overheating": 0.10,
            "position_drift": 0.10,
            "timeout": 0.10,
        },
        "rotate_wrist": {
            "joint_limit_exceeded": 0.25,
            "singularity_detected": 0.20,
            "motor_overheating": 0.20,
            "calibration_error": 0.15,
            "execution_error": 0.10,
            "timeout": 0.10,
        },
        "plan_trajectory": {
            "planning_timeout": 0.35,
            "workspace_violation": 0.20,
            "collision_detected": 0.15,
            "singularity_detected": 0.15,
            "network_failure": 0.10,
            "timeout": 0.05,
        },
        "visual_scan": {
            "vision_failure": 0.40,
            "object_not_found": 0.25,
            "sensor_malfunction": 0.15,
            "timeout": 0.10,
            "network_failure": 0.05,
            "calibration_error": 0.05,
        },
    }

    # Action primitives
    ACTIONS = [
        "move_to_position",
        "grasp_object",
        "release_object",
        "rotate_wrist",
        "open_gripper",
        "close_gripper",
        "visual_scan",
        "plan_trajectory",
        "execute_trajectory",
    ]

    # Object types
    OBJECTS = [
        "cup", "bottle", "box", "plate", "fork", "knife",
        "apple", "banana", "can", "towel", "book"
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.run_id = 0

    def generate_observation(self, timestep: int) -> Dict[str, Any]:
        """Generate a mock observation."""
        return {
            "timestep": timestep,
            "camera": {
                "rgb_shape": [224, 224, 3],
                "detected_objects": random.sample(self.OBJECTS, k=random.randint(1, 4))
            },
            "robot_state": {
                "joint_positions": [round(random.uniform(-3.14, 3.14), 3) for _ in range(7)],
                "gripper_open": random.choice([True, False]),
                "end_effector_pose": {
                    "position": [round(random.uniform(-0.5, 0.5), 3) for _ in range(3)],
                    "orientation": [round(random.uniform(-1, 1), 3) for _ in range(4)]
                }
            },
            "force_torque": [round(random.uniform(-5, 5), 2) for _ in range(6)]
        }

    def generate_action(self) -> Dict[str, Any]:
        """Generate a mock action."""
        action_type = random.choice(self.ACTIONS)
        action = {"type": action_type}

        if action_type in ["move_to_position", "execute_trajectory"]:
            action["target"] = [round(random.uniform(-0.5, 0.5), 3) for _ in range(3)]
        elif action_type == "grasp_object":
            action["object"] = random.choice(self.OBJECTS)
            action["force"] = round(random.uniform(5, 20), 1)
        elif action_type == "rotate_wrist":
            action["angle"] = round(random.uniform(-180, 180), 1)

        return action

    def generate_error(self, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a mock error with context."""
        error_messages = {
            # Original error types
            "gripper_slip": f"Object {context.get('object', 'unknown')} slipped from gripper",
            "object_not_found": f"Could not locate {context.get('object', 'target')} in scene",
            "collision_detected": "Collision detected during trajectory execution",
            "timeout": f"Action {context.get('action', 'unknown')} exceeded timeout (30s)",
            "joint_limit_exceeded": f"Joint {random.randint(0, 6)} exceeded limit",
            "force_threshold_exceeded": f"Force exceeded threshold: {round(random.uniform(20, 50), 1)}N",
            "vision_failure": "Failed to process camera feed",
            "grasp_failure": f"Failed to grasp {context.get('object', 'object')}",
            "planning_timeout": "Motion planning failed to find valid path",
            "execution_error": "Trajectory execution deviated from plan",

            # New error types
            "sensor_malfunction": f"Sensor {random.choice(['IMU', 'force', 'proximity', 'encoder'])} malfunction detected",
            "calibration_error": "Robot calibration out of tolerance",
            "network_failure": "Lost connection to control server",
            "motor_overheating": f"Motor {random.randint(1, 7)} temperature exceeded {random.randint(70, 90)}°C",
            "position_drift": f"Position drift detected: {round(random.uniform(0.5, 5.0), 2)}mm",
            "vacuum_loss": "Vacuum gripper pressure loss detected",
            "path_deviation": f"Path deviation exceeded {round(random.uniform(1, 10), 1)}mm threshold",
            "singularity_detected": "Robot approaching kinematic singularity",
            "workspace_violation": "Target position outside workspace limits",
            "emergency_stop": "Emergency stop triggered by safety system",
        }

        return {
            "type": error_type,
            "message": error_messages.get(error_type, "Unknown error"),
            "severity": random.choice(["warning", "error", "critical"]),
            "recoverable": random.choice([True, False])
        }

    def generate_run(self, run_length: int, failure_prob: float = 0.3) -> List[Dict[str, Any]]:
        """Generate a single robot run with possible failures."""
        self.run_id += 1
        run_logs = []
        base_time = datetime.now()

        failed = False
        failure_timestep = None
        failure_action = None

        # Decide if this run will fail and when
        if random.random() < failure_prob:
            failed = True
            failure_timestep = random.randint(run_length // 2, run_length - 1)

        for timestep in range(run_length):
            timestamp = base_time + timedelta(milliseconds=100 * timestep)

            log_entry = {
                "run_id": f"run_{self.run_id:05d}",
                "timestamp": timestamp.isoformat(),
                "timestep": timestep,
                "observation": self.generate_observation(timestep),
                "action": self.generate_action()
            }

            # Add error if this is the failure timestep
            if failed and timestep == failure_timestep:
                action = log_entry["action"]
                action_type = action["type"]

                # Select error type based on the action type
                if action_type in self.ACTION_ERROR_MAP:
                    error_map = self.ACTION_ERROR_MAP[action_type]
                    error_types = list(error_map.keys())
                    weights = list(error_map.values())
                    error_type = random.choices(error_types, weights=weights, k=1)[0]
                else:
                    # Fallback to uniform selection if action not in map
                    error_type = random.choice(["execution_error", "timeout", "emergency_stop"])

                context = {
                    "action": action_type,
                    "object": action.get("object", "unknown")
                }
                log_entry["error"] = self.generate_error(error_type, context)
                log_entry["status"] = "failed"
            else:
                log_entry["status"] = "success" if timestep < run_length - 1 else "completed"

            run_logs.append(log_entry)

            # Stop run if failed
            if failed and timestep == failure_timestep:
                break

        return run_logs

    def generate_dataset(self, num_runs: int, run_length: int = 50,
                        failure_prob: float = 0.3, output_file: str = "robot_logs.jsonl"):
        """Generate a dataset of robot runs."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Generating {num_runs} robot runs...")

        with open(output_path, 'w') as f:
            for i in range(num_runs):
                run = self.generate_run(run_length, failure_prob)
                for log_entry in run:
                    f.write(json.dumps(log_entry) + '\n')

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{num_runs} runs")

        print(f" Dataset saved to {output_path}")
        print(f"  Total runs: {num_runs}")
        print(f"  Expected failures: ~{int(num_runs * failure_prob)}")


def main():
    parser = argparse.ArgumentParser(description="Generate mock robot run logs")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="Number of robot runs to generate")
    parser.add_argument("--run-length", type=int, default=50,
                       help="Number of timesteps per run")
    parser.add_argument("--failure-prob", type=float, default=0.3,
                       help="Probability of run failure (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default="mockdata/robot_logs.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    generator = RobotLogGenerator(seed=args.seed)
    generator.generate_dataset(
        num_runs=args.num_runs,
        run_length=args.run_length,
        failure_prob=args.failure_prob,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
