"""VLM-based global frontier planner."""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Sequence

from utils.risk.frontier import (
    FrontierRiskReport,
    guard_frontier_assignments,
    risk_context_payload,
)

from .base import (
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
    random_goal_result,
)
from .co_ut import CostUtilityGlobalPlanner
from .errors import GPTResponseError


class GPTGlobalPlanner(GlobalPlanner):
    """Assign frontiers with the project's existing GPT-4o prompts."""

    name = "gpt"
    uses_shared_frontier_map = True
    risk_fallback_name = "co_ut"

    def __init__(
        self,
        *,
        chat_backend=None,
        prompts=None,
        cost_utility_lambda: float = 0.5,
    ) -> None:
        if chat_backend is None:
            from utils import chat_utils as chat_backend
        if prompts is None:
            import system_prompt as prompts
        self._chat = chat_backend
        self._prompts = prompts
        self._fallback = CostUtilityGlobalPlanner(cost_utility_lambda)

    @staticmethod
    def _emit_fallback(mode: str, error: GPTResponseError) -> None:
        print(
            "[gpt-fallback] "
            + json.dumps(
                {
                    "event": "gpt_fallback",
                    "mode": str(mode),
                    "fallback_planner": "co_ut",
                    "reason": error.reason,
                    "attempts": error.attempts,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if len(context.target_points) == 0 or context.local_step <= 0:
            return random_goal_result(context)

        candidate_maps = self._chat.get_all_candidate_maps(
            context.target_edge_map,
            context.top_view_map,
            context.poses,
        )
        message = self._chat.message_prepare(
            self._prompts.system_prompt,
            candidate_maps,
            context.goal_name,
            num_agents=context.num_agents,
        )
        try:
            raw_assignments = self._chat.chat_with_gpt4v(
                message,
                num_agents=context.num_agents,
                num_frontiers=len(context.target_points),
            )
        except GPTResponseError as error:
            self._emit_fallback("normal", error)
            return self._fallback.plan(context)
        assignments = {
            robot_id: int(
                raw_assignments[f"robot_{robot_id}"].split("_")[1]
            )
            for robot_id in range(context.num_agents)
        }
        return GlobalPlannerResult(
            goal_points=[
                goal_from_frontier(context, assignments[robot_id])
                for robot_id in range(context.num_agents)
            ],
            frontier_assignments=assignments,
        )

    def refine_risk_assignments(
        self,
        context: GlobalPlannerContext,
        reports: Sequence[FrontierRiskReport],
        fallback_assignments: Dict[int, Optional[int]],
        hard_risk_threshold: float,
    ) -> Dict[int, Optional[int]]:
        if len(context.target_points) == 0 or context.local_step <= 0:
            return fallback_assignments

        candidate_maps = self._chat.get_all_candidate_maps(
            context.target_edge_map,
            context.top_view_map,
            context.poses,
        )
        message = self._chat.risk_message_prepare(
            self._prompts.risk_prompt,
            candidate_maps,
            context.goal_name,
            risk_context=risk_context_payload(reports),
            num_agents=context.num_agents,
        )
        try:
            raw_assignments = self._chat.chat_with_gpt4v(
                message,
                num_agents=context.num_agents,
                num_frontiers=len(context.target_points),
            )
        except GPTResponseError as error:
            self._emit_fallback("risk", error)
            return fallback_assignments
        guarded = guard_frontier_assignments(
            raw_assignments,
            reports,
            fallback_assignments=fallback_assignments,
            expected_robot_ids=range(context.num_agents),
            hard_risk_threshold=hard_risk_threshold,
        )
        if guarded.rejected:
            logging.warning(
                "risk guard replaced VLM frontier choices: %s",
                guarded.rejected,
            )
        return guarded.assignments
