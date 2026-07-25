import importlib

# Verified against agentdojo==0.1.35 source (agentdojo/base_tasks.py):
#   class BaseUserTask: ID: str; PROMPT: str; GROUND_TRUTH_OUTPUT: str = ""
# Task prompts live on the *uppercase* class attribute `PROMPT`, not on any
# lowercase attribute. The original lowercase-only lookup below silently
# matched nothing and made `load_agentdojo_tasks` raise "Could not extract
# AgentDojo tasks" on every run. `PROMPT`/`GOAL` are checked first because
# they are the real, current API; the lowercase names are kept only as a
# defensive fallback in case a future agentdojo release renames the field.
_ATTR_CANDIDATES = ("PROMPT", "GOAL", "prompt", "instruction", "user_prompt", "text")


def _task_text(obj):
    for name in _ATTR_CANDIDATES:
        value = getattr(obj, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(obj, dict):
        for name in _ATTR_CANDIDATES:
            value = obj.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_agentdojo_tasks(limit=97):
    """Load AgentDojo user-task prompts across the four default suites.

    Primary path uses the documented, verified 0.1.35 API:
        agentdojo.task_suite.load_suites.get_suite(benchmark_version, suite_name)
        -> TaskSuite.user_tasks -> dict[str, BaseUserTask] (already instantiated)
        -> task.PROMPT

    A set of alternative import paths / call signatures is retried in case a
    different agentdojo version is installed, so this keeps working across
    minor API drifts rather than hard-failing on the first mismatch.
    """
    candidates = [
        ("agentdojo.task_suite.load_suites", "get_suite"),
        ("agentdojo.task_suite", "get_suite"),
        ("agentdojo.benchmark", "get_suite"),
    ]
    suites = ("workspace", "banking", "travel", "slack")
    benchmark_versions = ("v1.2.2", "v1.2.1", "v1.2", "v1.1.1", "v1.1", "v1")
    rows = []
    errors = []
    seen_ids = set()

    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            loader = getattr(module, function_name)
        except Exception as exc:
            errors.append(f"{module_name}.{function_name}: {exc}")
            continue

        for suite_name in suites:
            suite = None
            attempts = []
            for version in benchmark_versions:
                attempts.append({"benchmark_version": version, "suite_name": suite_name})
            attempts += [
                {"suite_name": suite_name},
                {"name": suite_name},
            ]
            for kwargs in attempts:
                try:
                    suite = loader(**kwargs)
                    break
                except Exception as exc:
                    errors.append(f"{module_name}.{function_name}({kwargs}): {exc}")
                    continue
            if suite is None:
                continue

            for attr in ("user_tasks", "tasks"):
                collection = getattr(suite, attr, None)
                if collection is None:
                    continue
                items = collection.items() if isinstance(collection, dict) else enumerate(collection)
                for task_id, task in items:
                    text = _task_text(task)
                    if not text:
                        continue
                    source_id = f"{suite_name}:{task_id}"
                    if source_id in seen_ids:
                        continue
                    seen_ids.add(source_id)
                    rows.append({
                        "source_id": source_id,
                        "task": text,
                        "suite": suite_name,
                    })
                    if len(rows) >= limit:
                        return rows

    if not rows:
        raise RuntimeError(
            "Could not extract AgentDojo tasks from the installed agentdojo "
            "package. The package API may have changed since 0.1.35. "
            "Diagnostics (last 10): " + " | ".join(errors[-10:])
        )
    return rows
