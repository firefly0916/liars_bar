from __future__ import annotations


def _step_tag(step: int) -> str:
    return f"step-{int(step):06d}"


def build_checkpoint_schedule(
    *,
    max_step: int,
    dense_until_step: int,
    dense_interval: int,
    sparse_interval: int,
    include_final: bool = True,
) -> list[str]:
    max_step = max(1, int(max_step))
    dense_until_step = max(0, min(int(dense_until_step), max_step))
    dense_interval = max(1, int(dense_interval))
    sparse_interval = max(1, int(sparse_interval))

    steps: list[int] = []
    current = dense_interval
    while current <= dense_until_step:
        steps.append(current)
        current += dense_interval

    sparse_start = dense_until_step + sparse_interval
    current = sparse_start
    while current <= max_step:
        steps.append(current)
        current += sparse_interval

    if max_step not in steps:
        steps.append(max_step)

    tags = [_step_tag(step) for step in sorted(set(steps))]
    if include_final:
        tags.append("final")
    return tags
