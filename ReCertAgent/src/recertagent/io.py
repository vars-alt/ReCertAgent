from pathlib import Path
from typing import Iterable, TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def write_jsonl(path, rows: Iterable[BaseModel]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")

def read_jsonl(path, cls: Type[T]) -> list[T]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(cls.model_validate_json(line))
    return out
