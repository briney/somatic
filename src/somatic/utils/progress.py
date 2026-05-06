"""Rich-based progress bars for training, evaluation, and encoding.

The single ``ProgressManager`` owns one persistent training task and any
number of eval-cycle tasks. Eval tasks added during one cycle remain
visible (frozen at 100% with their final elapsed time) until
``start_eval_cycle()`` is called again, at which point they are removed
and replaced by the next cycle's bars.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from accelerate import Accelerator


_BAR_WIDTH = 40


def _make_columns() -> list:
    """Build the fixed column layout used everywhere."""
    return [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=_BAR_WIDTH),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
    ]


def make_progress(disable: bool = False) -> Progress:
    """Build a Rich ``Progress`` with the project's standard column layout.

    Args:
        disable: If True, the progress bar renders nothing (used on
            non-main ranks in distributed training).

    Returns:
        A configured ``Progress`` instance. The caller is responsible for
        starting and stopping it (``Progress`` is itself a context manager).
    """
    console = Console(file=sys.stdout, force_terminal=None)
    return Progress(
        *_make_columns(),
        console=console,
        transient=False,
        expand=False,
        disable=disable,
    )


class _EvalTaskHandle:
    """Lightweight handle returned by ``ProgressManager.eval_task``."""

    def __init__(self, progress: Progress | None, task_id: TaskID | None) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int = 1) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, n)


class ProgressManager:
    """Manages the training progress bar and per-eval-cycle sub-bars.

    A single instance is owned by the trainer for the duration of
    ``train()``. The Evaluator is given a reference and uses
    ``eval_task()`` to add sub-bars for each named eval / region eval /
    per-position pass. ``start_eval_cycle()`` is called by the trainer
    immediately before each eval cycle to clear the previous cycle's bars.
    """

    def __init__(
        self,
        accelerator: "Accelerator | None" = None,
        disable: bool = False,
    ) -> None:
        if accelerator is not None and not accelerator.is_local_main_process:
            disable = True
        self._progress = make_progress(disable=disable)
        self._disabled = disable
        self._train_task_id: TaskID | None = None
        self._eval_task_ids: list[TaskID] = []
        self._started = False

    def __enter__(self) -> "ProgressManager":
        self._progress.start()
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._started:
            self._progress.stop()
            self._started = False

    def train_task(self, total: int, start: int = 0) -> TaskID:
        """Add the persistent training bar. Call once."""
        if self._disabled:
            return TaskID(-1)
        self._train_task_id = self._progress.add_task(
            "Training", total=total, completed=start
        )
        return self._train_task_id

    def advance_train(self, n: int = 1) -> None:
        if self._train_task_id is not None and not self._disabled:
            self._progress.advance(self._train_task_id, n)

    def start_eval_cycle(self) -> None:
        """Clear all eval-cycle tasks from the previous cycle.

        The training task is preserved.
        """
        if self._disabled:
            return
        for task_id in self._eval_task_ids:
            try:
                self._progress.remove_task(task_id)
            except KeyError:
                pass
        self._eval_task_ids = []

    @contextmanager
    def eval_task(self, description: str, total: int) -> Iterator[_EvalTaskHandle]:
        """Add an eval sub-bar that stays visible at 100% on exit.

        The task ID is tracked so the next ``start_eval_cycle()`` removes it.
        """
        if self._disabled:
            yield _EvalTaskHandle(None, None)
            return

        task_id = self._progress.add_task(description, total=max(total, 0))
        self._eval_task_ids.append(task_id)
        try:
            yield _EvalTaskHandle(self._progress, task_id)
        finally:
            # Snap to 100% so the final time is what the user sees.
            try:
                task = self._progress.tasks[
                    [t.id for t in self._progress.tasks].index(task_id)
                ]
                if task.total is not None and task.completed < task.total:
                    self._progress.update(task_id, completed=task.total)
            except (ValueError, IndexError):
                pass

    @staticmethod
    @contextmanager
    def standalone_eval_task(
        description: str, total: int, disable: bool = False
    ) -> Iterator[_EvalTaskHandle]:
        """One-shot progress bar for callers without a ProgressManager.

        Used by the encode CLI and by the Evaluator when invoked outside
        of a training run.
        """
        if disable:
            yield _EvalTaskHandle(None, None)
            return

        progress = make_progress(disable=False)
        progress.start()
        try:
            task_id = progress.add_task(description, total=max(total, 0))
            yield _EvalTaskHandle(progress, task_id)
            # Snap to 100% on clean exit.
            task = progress.tasks[[t.id for t in progress.tasks].index(task_id)]
            if task.total is not None and task.completed < task.total:
                progress.update(task_id, completed=task.total)
        finally:
            progress.stop()
