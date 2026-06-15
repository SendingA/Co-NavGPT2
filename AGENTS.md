# Global Bot Instructions

These rules apply to all projects in this repository unless a project-specific instruction explicitly overrides them.

## 0) Post-Task Follow-up

When the **top-level** user request (including all sub-tasks) is fully complete:

1. Call the ask-questions interface **once** with two options:
   - Option 1: "No more tasks" (end conversation turn)
   - Option 2: "Enter the next task"
2. If the user selects Option 2 and provides a new task, execute it fully, then repeat step 1.
3. If the user selects Option 1, stop.
4. If the ask-questions call fails or receives no response, treat it as Option 1 (stop gracefully).

**Do NOT** call the ask-questions interface after individual sub-steps within a multi-step task. Only call it after the entire requested work is done.

subagent严禁调用提问！需要主agent给subagent强调。只有当前任务完全完成需要用户检查时主agent才可调用


## 1) Task Planning And TODO Management

1. Every non-trivial request must be decomposed into a detailed TODO list before implementation.
2. Store TODOs as JSON in the target project under:
    - `<project-root>/todo/todo.json`
     - `project-root` means the top folder of that project (for example `idea2html2video`), not any nested path like `src/...`.
     - Never place TODO tracking files under `src/`.
3. If the file does not exist, create it.
4. Update TODO status frequently during execution:
    - At least once after each completed subtask.
    - At least once every 20 minutes during long tasks.
    - Immediately when blockers appear.
5. Keep TODO items granular and actionable (small tasks that can be validated independently).

### Required TODO JSON Schema

```json
{
  "meta": {
     "project": "string",
     "updated_at": "ISO-8601 datetime",
     "owner": "copilot"
  },
  "items": [
     {
        "id": "T001",
        "title": "short action title",
        "status": "todo | in_progress | done | blocked",
        "priority": "P0 | P1 | P2",
        "depends_on": ["T000"],
        "acceptance": "clear completion criteria",
        "notes": "optional",
        "last_updated": "ISO-8601 datetime"
     }
  ]
}
```

## 2) File Placement Conventions

Use these default locations when creating or moving files:

1. `src/`
    - Production source code and runtime modules.
2. `scripts/`
    - Automation scripts, one-off tooling, migration helpers.
3. `tests/`
    - Unit/integration/e2e tests mirroring `src/` structure.
4. `docs/`
    - User/developer documentation, architecture notes, ADRs.
5. `todo/`
    - Task tracking files (must include `todo.json`).
6. `assets/`
    - Static assets used by the project (images, fonts, templates).
7. `data/`
    - Input/reference datasets (`data/raw`, `data/processed`).
8. `outputs/`
    - Generated artifacts (reports, exported media, temporary generated files).
9. `ref/`
    - External references, experiments, or legacy material. Do not place core workflow code here.
10. Project root
    - Only for essential config and entry files (for example `package.json`, `pyproject.toml`, `README.md`, `.gitignore`).

## 3) Mandatory Validation After Every Change

After writing or modifying files, always validate the change.

1. Code changes:
    - Run relevant tests.
    - Run lint/type checks if configured.
    - Run a minimal smoke execution path if tests are missing.
2. Script changes:
    - Execute the script with representative arguments.
    - Confirm expected output files are created and readable.
3. Template/UI changes:
    - Render once and verify no runtime errors.
    - Verify critical visual fields (title/body/metadata) are readable.
4. Documentation/instruction changes:
    - Verify file paths, commands, and referenced filenames are correct.

If validation cannot be completed, explicitly record:

1. What could not be tested.
2. Why it could not be tested.
3. What should be tested next.

## 4) Completion Checklist

A task is complete only when all are true:

1. TODO items are updated in `todo/todo.json`.
2. Files are placed according to conventions above.
3. Validation has run and results are recorded.
4. Outputs/artifacts are in the correct `outputs/` location.
5. Any blockers or follow-up actions are added as TODO items.

## 5) Update Discipline

1. Prefer incremental commits/changes over large monolithic edits.
2. Keep public interfaces stable unless explicitly requested.
3. Avoid silent behavior changes; document intent in TODO notes.
4. When moving files, update all related references and commands.