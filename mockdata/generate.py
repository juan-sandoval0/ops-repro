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

    # Common error types in robotics systems
    ERROR_TYPES = [
        "gripper_slip",
        "object_not_found",
        "collision_detected",
        "timeout",
        "joint_limit_exceeded",
        "force_threshold_exceeded",
        "vision_failure",
        "grasp_failure",
        "planning_timeout",
        "execution_error",
    ]

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
        error_type = None

        # Decide if this run will fail and when
        if random.random() < failure_prob:
            failed = True
            failure_timestep = random.randint(run_length // 2, run_length - 1)
            error_type = random.choice(self.ERROR_TYPES)

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
                context = {
                    "action": action["type"],
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
