from recertagent.adapters.agentdojo import _task_text


class _FakeUserTask:
    """Mimics agentdojo.base_tasks.BaseUserTask's real shape: the prompt
    lives on the uppercase `PROMPT` class attribute, not any lowercase
    name. This is a regression test for the bug where `_task_text` only
    checked lowercase attributes and therefore extracted zero tasks from
    every real AgentDojo suite."""

    ID = "user_task_0"
    PROMPT = "Send the quarterly report to the finance channel."


def test_task_text_reads_uppercase_prompt_attribute():
    assert _task_text(_FakeUserTask()) == "Send the quarterly report to the finance channel."


def test_task_text_ignores_blank_prompt():
    class _Blank:
        PROMPT = "   "

    assert _task_text(_Blank()) is None


def test_task_text_falls_back_to_dict_lookup():
    assert _task_text({"prompt": "Book a flight to Zurich."}) == "Book a flight to Zurich."


def test_task_text_returns_none_when_nothing_matches():
    class _Empty:
        pass

    assert _task_text(_Empty()) is None
