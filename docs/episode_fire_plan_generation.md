# 从任意 ObjectNav episode 生成 route-contrast Fire Plan

`scripts/generate_episode_fire_plan.py` 把 person/bed 成功案例中的人工步骤
整理成一个确定性的 episode 级搜索器。目标不是“在路径附近放火”，而是构造并
验证同一任务上的反事实：不使用风险的最短路较短但危险，risk-aware 路径稍长、
拓扑不同且安全，同时目标成功区域仍可到达。

## 成功案例中真正可迁移的部分

1. 固定 episode，而不是先放火再寻找有利的任务。episode 由 scene、
   `episode_id`、`object_category` 和原始起点共同确定。
2. 终点使用数据集原生 `goals_by_category[*].view_points`。不创建另一个目标坐标，
   因而 Habitat 的 DistanceToGoal、STOP、Success 和 SPL 语义不变。
3. 在无风险 obstacle map 上计算 blind shortest path。长度不足 4 m 的 episode
   默认拒绝，因为它通常没有足够空间形成可读的分岔。
4. 只考虑当前楼层、非结构、`flammability >= 0.4`、高度不超过 1 m，而且投影到
   blind route 附近的真实语义对象。
5. 火源不能太靠近起点或目标。person 默认要求距离 person centre 至少 3.0 m、
   距离任一原生 VIEW_POINT 至少 2.5 m；其他类别默认分别为 1.5 m 和 1.2 m。
   默认也禁止点燃与目标相同类别的对象。
6. 对候选对象组合进行搜索。三处火源的风险使用逐格最大值合并，hard unsafe
   使用集合并集；每个最终组合都重新计算 blind 和 aware 路径，避免“单火源验证
   通过，补上另外两处火后却破坏安全绕路”的问题。
7. 只有联合结果满足 route-contrast contract 才会写 plan。stable 默认要求：

   - blind path max risk `>= 0.60`；
   - aware path max risk `<= 0.35`，且不穿过 hard unsafe；
   - exposure reduction `>= 0.70`；
   - detour ratio 在 `[1.15, 1.80]`；
   - path divergence `>= 0.20`；
   - 所有原生目标 VIEW_POINTS 均不进入 hard unsafe，代理风险不超过 `0.35`。

8. 合格组合按 exposure reduction、path divergence、适中的 detour ratio、
   blind/aware 风险差进行评分；同分时优先选择沿短路分布更开的火源。
9. plan 对完整 payload 做内容寻址，任何火源、传播参数或 curation 变化都会产生
   新 `plan_id`，不会静默覆盖旧实验。
10. 代理搜索只证明几何拓扑。正式实验必须 bake `timeline.npz`，并在运行时范围内
    使用真实 flame/smoke/temperature 投影重新验证路径和目标保护。

## 通用命令

HM3D shard 中的 `episode_id` 经常在不同 category 之间重复，因此建议始终显式给出
`--object-category`：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/generate_episode_fire_plan.py \
  --source-shard data/datasets/objectnav_hm3d_v2/val/content/<SCENE>.json.gz \
  --episode-id <EPISODE_ID> \
  --object-category <CATEGORY> \
  --profile stable \
  --source-count 3 \
  --write-dataset
```

默认输出：

```text
outputs/fire_episode_plans/<scene>_ep_<id>_<category>_stable/
├── episode_fire_plan.json
└── surrogate_route_contrast.png

scenes/<scene>/plans/<content-addressed-plan-id>.json
data/processed/fire_route_scenarios/<plan-id>/val.json.gz
```

先检查 `episode_fire_plan.json` 中的 `accepted`、`selected_sources`、
`combined_surrogate`、`protection` 和 `validation_status`。未 bake 时状态是
`surrogate_only`；只有 `actual_timeline_passed` 才表示真实传播也通过。确认代理图
符合预期后再 bake：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/generate_episode_fire_plan.py \
  --source-shard data/datasets/objectnav_hm3d_v2/val/content/<SCENE>.json.gz \
  --episode-id <EPISODE_ID> \
  --object-category <CATEGORY> \
  --profile stable \
  --source-count 3 \
  --write-dataset \
  --bake
```

`--bake` 会写入标准目录
`outputs/fire_world/<scene>/<plan-id>/timeline.npz`，并默认在 40、60、100 秒
重新验证 route contrast 和所有目标 VIEW_POINTS。已有 timeline 可用
`--validate-existing` 复验而不重新传播。

## Person 与 bed 示例

Person episode 10 使用稍大的实火源半径，以匹配已验证案例：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/generate_episode_fire_plan.py \
  --source-shard data/datasets/objectnav_hm3d_person_v1/val/content/Nfvxx8J5NCo.json.gz \
  --episode-id 10 \
  --object-category person \
  --source-count 3 \
  --source-radius-m 0.78 \
  --write-dataset
```

Bed episode 5 使用 stable profile 默认半径：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/generate_episode_fire_plan.py \
  --source-shard data/datasets/objectnav_hm3d_v2/val/content/Nfvxx8J5NCo.json.gz \
  --episode-id 5 \
  --object-category bed \
  --source-count 3 \
  --write-dataset
```

## 搜索失败如何解释

失败仍会写 `episode_fire_plan.json`。`search.rejection_counts` 会给出原因，常见项：

- `insufficient_source_candidates`：最短路附近没有足够的低矮可燃对象；
- `blind_path_not_dangerous`：火源没有真正影响最短路；
- `aware_path_still_dangerous` 或 `aware_path_crosses_hard_unsafe`：火区过大；
- `detour_too_short`：只是轻微侧移，不能明显体现 risk awareness；
- `detour_too_long` 或 `no_route`：火灾切断了任务；
- `goal_risk_too_high`：火源覆盖了原生成功区域；
- `paths_not_topologically_distinct`：两条路线视觉上仍高度重合。

优先换 episode 或扩大 `--max-source-candidates`，不要首先放松安全阈值。若只缺少
语义对象，可谨慎增大 `--max-source-to-blind-route-m`；若实际 timeline 太弱，可在
保持 `floor_max_spread_radius_m` 有界的前提下调整 `--source-radius-m` 并重新 bake。
