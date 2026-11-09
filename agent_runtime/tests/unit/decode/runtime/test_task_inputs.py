"""``decode run`` task resolution (APICookbook fork: an explicit task is required)."""

import pytest

from decode.runtime.task_inputs import NO_TASK_MESSAGE, WorkerTaskInputError, resolve_task_and_model


def test_an_explicit_task_and_model_are_cleaned():
    assert resolve_task_and_model("  do it ", " m ") == ("do it", "m")
    assert resolve_task_and_model("do it", "  ") == ("do it", None)


@pytest.mark.parametrize("task", [None, "", "   "])
def test_no_task_is_refused(task):
    with pytest.raises(WorkerTaskInputError, match="needs a TASK"):
        resolve_task_and_model(task, None)
    assert "decode run" in NO_TASK_MESSAGE
