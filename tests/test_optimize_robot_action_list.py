import math
import unittest

from scripts.optimize_robot_action_list import (
    _distance_xz,
    optimize_payload,
)


def _action(agent_id, position, name="move_forward", risk=0.0):
    return {
        "agent_id": agent_id,
        "action": 1 if name == "move_forward" else 0,
        "action_name": name,
        "position_after": list(position),
        "risk_after": risk,
        "hard_unsafe_after": False,
    }


def _payload(agent_paths, terminal_names=None):
    terminal_names = terminal_names or ["stop"] * len(agent_paths)
    steps = []
    for step in range(max(len(path) for path in agent_paths)):
        actions = []
        for agent_id, path in enumerate(agent_paths):
            position = path[min(step, len(path) - 1)]
            name = (
                terminal_names[agent_id]
                if step == max(len(path) for path in agent_paths) - 1
                else "move_forward"
            )
            actions.append(_action(
                agent_id,
                position,
                name=name,
                risk=0.01 * step,
            ))
        steps.append({
            "step": step,
            "t_sim_s": 0.4 * step,
            "wall_time_s": 0.1,
            "actions": actions,
        })
    return {
        "episode_id": "test",
        "scene_id": "scene",
        "fire_plan_id": "plan",
        "planner_source": "oracle",
        "num_agents": len(agent_paths),
        "num_steps": len(steps),
        "steps": steps,
    }


class RobotActionOptimizationTests(unittest.TestCase):
    def test_straight_path_drops_stationary_poses_and_caps_spacing(self):
        path = [
            [0.0, 0.2, 0.0],
            [0.0, 0.2, 0.0],
            [0.25, 0.2, 0.0],
            [0.50, 0.2, 0.0],
            [0.75, 0.2, 0.0],
            [1.00, 0.2, 0.0],
            [1.00, 0.2, 0.0],
        ]
        result = optimize_payload(
            _payload([path]),
            source_action_list="source.json",
            path_deviation_m=0.10,
            max_waypoint_spacing_m=0.60,
        )
        agent = result["agents"]["0"]
        route = agent["route_waypoints"]
        self.assertEqual(route[0]["position_xyz_m"], path[0])
        self.assertEqual(route[-1]["position_xyz_m"], path[-1])
        self.assertLessEqual(agent["waypoint_spacing"]["maximum_m"], 0.60 + 1e-9)
        self.assertLess(agent["execution_command_count"], len(path))
        self.assertEqual(agent["terminal_command"], "STOP")
        self.assertEqual(agent["commands"][-1]["command"], "STOP")

    def test_bounded_simplification_preserves_a_right_angle(self):
        path = [
            [0.0, 0.2, 0.0],
            [0.4, 0.2, 0.0],
            [0.8, 0.2, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.2, 0.4],
            [1.0, 0.2, 0.8],
            [1.0, 0.2, 1.0],
        ]
        result = optimize_payload(
            _payload([path]),
            source_action_list="source.json",
            path_deviation_m=0.10,
            max_waypoint_spacing_m=0.90,
        )
        agent = result["agents"]["0"]
        route_positions = [
            item["position_xyz_m"] for item in agent["route_waypoints"]
        ]
        self.assertTrue(any(
            _distance_xz(position, [1.0, 0.2, 0.0]) <= 1e-9
            for position in route_positions
        ))
        self.assertLessEqual(agent["maximum_source_path_deviation_m"], 0.10 + 1e-9)
        self.assertAlmostEqual(agent["optimized_path_length_m"], 2.0, places=6)

    def test_two_agent_output_has_explicit_world_frame_and_hold(self):
        first = [[0.0, 0.2, 0.0], [0.0, 0.2, 1.0]]
        second = [[2.0, 0.2, 0.0], [2.5, 0.2, 0.0]]
        result = optimize_payload(
            _payload([first, second], terminal_names=["stop", "turn_left"]),
            source_action_list="source.json",
            max_waypoint_spacing_m=0.75,
        )
        self.assertEqual(result["coordinate_frame"]["name"], "habitat_world_xyz_m")
        self.assertFalse(result["coordinate_frame"]["open3d_coordinates_used"])
        self.assertIn("column=x, row=z", result["coordinate_frame"]["image_overlay"])
        self.assertEqual(result["agents"]["0"]["terminal_command"], "STOP")
        self.assertEqual(result["agents"]["1"]["terminal_command"], "HOLD")
        heading = result["agents"]["1"]["route_waypoints"][0][
            "heading_hint_rad"
        ]
        self.assertAlmostEqual(heading, math.pi / 2.0)
        self.assertGreater(
            result["agents"]["0"]["route_waypoints"][-1][
                "max_source_risk_since_previous_waypoint"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
