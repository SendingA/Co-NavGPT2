import cv2
import numpy as np
import skfmm
import skimage
from numpy import around, array, ma


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2 \
               and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 > \
                    (step_size - 1) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    """Fast-marching planner with optional, spatially varying risk cost.

    ``risk_map`` is expected to be a normalized ``[0, 1]`` grid aligned with
    ``traversible``.  When ``risk_alpha`` is positive, the Eikonal speed is
    ``1 / (1 + risk_alpha * risk_map)``.  With no risk inputs (the historical
    default), the implementation continues to use ``skfmm.distance``.

    ``hard_unsafe_mask`` removes cells from the traversible domain.  Goal
    cells are explicitly admitted by ``set_goal``/``set_multi_goal`` so a
    dilated safety mask cannot make a valid navigation goal numerically
    unreachable.  Call :meth:`prepare_emergency_escape` before setting a goal
    when an agent starts inside a newly unsafe region.
    """

    def __init__(self, traversible, scale=1, step_size=5, risk_map=None,
                 risk_alpha=0.0, hard_unsafe_mask=None):
        self.scale = scale
        self.step_size = step_size
        self.risk_alpha = max(0.0, float(risk_alpha))

        traversible = np.asarray(traversible)
        original_shape = traversible.shape
        if traversible.ndim != 2:
            raise ValueError("traversible must be a 2-D grid")

        risk_map = self._validate_optional_grid(
            risk_map, original_shape, "risk_map"
        )
        hard_unsafe_mask = self._validate_optional_grid(
            hard_unsafe_mask, original_shape, "hard_unsafe_mask"
        )

        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
            if risk_map is not None:
                risk_map = cv2.resize(
                    risk_map.astype(np.float32),
                    (self.traversible.shape[1], self.traversible.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            if hard_unsafe_mask is not None:
                hard_unsafe_mask = cv2.resize(
                    hard_unsafe_mask.astype(np.uint8),
                    (self.traversible.shape[1], self.traversible.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        else:
            self.traversible = traversible

        if risk_map is None:
            self.risk_map = None
        else:
            self.risk_map = np.nan_to_num(
                np.asarray(risk_map, dtype=np.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            self.risk_map = np.clip(self.risk_map, 0.0, 1.0)

        if hard_unsafe_mask is None:
            self.hard_unsafe_mask = np.zeros(
                self.traversible.shape, dtype=bool
            )
        else:
            self.hard_unsafe_mask = np.asarray(
                hard_unsafe_mask, dtype=bool
            ).copy()

        # Keep the obstacle-only domain so an agent engulfed by a newly
        # updated hazard can be given a one-way escape corridor.
        self.base_traversible = np.asarray(self.traversible).copy()
        self.base_traversible_mask = self.base_traversible != 0
        if np.any(self.hard_unsafe_mask):
            self.traversible = (
                self.base_traversible_mask & ~self.hard_unsafe_mask
            ).astype(np.float32)
        else:
            # Retain the original values/dtype in the risk-disabled case.
            self.traversible = self.base_traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.fmm_dist = None
        self.around = np.zeros(traversible.shape)

        self._rebuild_around()

    @staticmethod
    def _validate_optional_grid(grid, shape, name):
        if grid is None:
            return None
        array_grid = np.asarray(grid)
        if array_grid.shape != shape:
            raise ValueError(
                "{} shape {} does not match traversible shape {}".format(
                    name, array_grid.shape, shape
                )
            )
        return array_grid

    @property
    def risk_enabled(self):
        return (
            (self.risk_map is not None and self.risk_alpha > 0.0)
            or np.any(self.hard_unsafe_mask)
        )

    def _rebuild_around(self):
        self.around = np.zeros(self.traversible.shape)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
        self.traversible_around = cv2.dilate((1-self.traversible*1).astype('uint8'), kernel)
        traversible_around_ma = ma.masked_values(self.traversible_around*1, 0)
        traversible_around_ma[self.traversible==0] = 0
        if np.any(traversible_around_ma == 0):
            dd = skfmm.distance(traversible_around_ma, dx=1)
            dd = (np.max(dd) - dd)
            dd = ma.filled(dd, 0)
            self.around = dd

    @staticmethod
    def _filled_distance(distance, fallback):
        compressed = ma.asarray(distance).compressed()
        if compressed.size == 0:
            fill_value = float(fallback)
        else:
            finite = compressed[np.isfinite(compressed)]
            fill_value = (
                float(np.max(finite)) + 1.0
                if finite.size
                else float(fallback)
            )
        return ma.filled(distance, fill_value)

    def _solve_distance(self, traversible_ma):
        if self.risk_map is not None and self.risk_alpha > 0.0:
            speed = 1.0 / (1.0 + self.risk_alpha * self.risk_map)
            speed_ma = ma.array(
                speed,
                mask=ma.getmaskarray(traversible_ma),
                copy=False,
            )
            distance = skfmm.travel_time(
                traversible_ma, speed_ma, dx=1
            )
        else:
            distance = skfmm.distance(traversible_ma, dx=1)
        return self._filled_distance(
            distance, fallback=np.prod(self.traversible.shape)
        )

    def prepare_emergency_escape(self, start):
        """Open the shortest obstacle-free corridor out of a hard hazard.

        Returns a single-cell goal map when ``start`` lies inside the hard
        mask, otherwise ``None``.  The exception is deliberately local to the
        escape corridor; all other hard-unsafe cells remain blocked.
        """
        sx = int(np.clip(start[0], 0, self.traversible.shape[0] - 1))
        sy = int(np.clip(start[1], 0, self.traversible.shape[1] - 1))
        if not self.hard_unsafe_mask[sx, sy]:
            return None

        from collections import deque

        queue = deque([(sx, sy)])
        parents = {(sx, sy): None}
        escape = None
        while queue:
            cell = queue.popleft()
            if (not self.hard_unsafe_mask[cell]
                    and self.base_traversible_mask[cell]):
                escape = cell
                break
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < self.traversible.shape[0] and
                        0 <= nxt[1] < self.traversible.shape[1]):
                    continue
                if nxt in parents or not self.base_traversible_mask[nxt]:
                    continue
                parents[nxt] = cell
                queue.append(nxt)

        if escape is None:
            # No obstacle-free way out exists. Admit only the current cell so
            # downstream FMM calls remain well-defined and request a replan.
            self.hard_unsafe_mask[sx, sy] = False
            self.traversible[sx, sy] = 1.0
            self._rebuild_around()
            return np.zeros(self.traversible.shape, dtype=np.uint8)

        cell = escape
        while cell is not None:
            self.hard_unsafe_mask[cell] = False
            cell = parents[cell]
        self.traversible = (
            self.base_traversible_mask & ~self.hard_unsafe_mask
        ).astype(np.float32)
        self._rebuild_around()

        escape_goal = np.zeros(self.traversible.shape, dtype=np.uint8)
        escape_goal[escape] = 1
        return escape_goal


    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
            int(goal[1] / (self.scale * 1.))

        goal_x = int(np.clip(goal_x, 0, self.traversible.shape[0] - 1))
        goal_y = int(np.clip(goal_y, 0, self.traversible.shape[1] - 1))

        if self.traversible[goal_x, goal_y] == 0. and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        self.traversible[goal_x, goal_y] = 1.0
        self.hard_unsafe_mask[goal_x, goal_y] = False
        traversible_ma[goal_x, goal_y] = 0
        self.fmm_dist = self._solve_distance(traversible_ma)
        return

    def set_multi_goal(self, goal_map):
        goal_map = np.asarray(goal_map)
        if goal_map.shape != self.traversible.shape:
            raise ValueError(
                "goal_map shape {} does not match traversible shape {}".format(
                    goal_map.shape, self.traversible.shape
                )
            )
        traversible_ma = ma.masked_values(self.traversible * 1, 0) #mask掉障碍物层(0),只剩空白区域
        goal_cells = goal_map == 1
        if not np.any(goal_cells):
            raise ValueError(
                "goal_map must contain at least one goal cell before FMM solve"
            )
        self.traversible[goal_cells] = 1.0
        self.hard_unsafe_mask[goal_cells] = False
        traversible_ma[goal_cells] = 0 # 除去障碍物层的地图，目标为0，其他为1
        self.fmm_dist = self._solve_distance(traversible_ma)
        self.around[goal_cells] = 0
        return

    def get_short_term_goal(self, state):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]
        self.fmm_dist += self.around
        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
        
        dist_around = np.pad(self.around, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset_around = dist_around[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if subset[self.du, self.du] < 0.25 * 100 / 5.:  # 25cm
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        # remove the imfluence of the around map
        if subset[stg_x, stg_y]+subset_around[self.du, self.du] > -0.0001:
            replan = True
        else:
            replan = False

        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    def _find_nearest_goal(self, goal):
        traversible = skimage.morphology.binary_dilation(
            np.zeros(self.traversible.shape),
            skimage.morphology.disk(2)) != True
        traversible = traversible * 1.
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
