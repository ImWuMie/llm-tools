from __future__ import annotations

from train import resolve_eval_strategy, split_records


def test_split_records_holds_out_at_least_one() -> None:
    records = [{"text": str(i)} for i in range(10)]
    train, ev = split_records(records, 0.1, seed=42)
    assert len(train) == 9
    assert len(ev) == 1
    assert {row["text"] for row in train + ev} == {str(i) for i in range(10)}


def test_split_records_disabled() -> None:
    records = [{"text": "a"}, {"text": "b"}]
    train, ev = split_records(records, 0.0, seed=1)
    assert train == records
    assert ev == []


def test_split_records_tiny_set_keeps_train() -> None:
    records = [{"text": "a"}, {"text": "b"}]
    train, ev = split_records(records, 0.5, seed=0)
    assert len(train) == 1
    assert len(ev) == 1


def test_resolve_eval_strategy() -> None:
    assert resolve_eval_strategy({}, has_eval=False) == ("no", 0)
    assert resolve_eval_strategy({"eval_strategy": "no"}, has_eval=True) == ("no", 0)
    strategy, steps = resolve_eval_strategy({"logging_steps": 20}, has_eval=True)
    assert strategy == "steps"
    assert steps == 20
    strategy, steps = resolve_eval_strategy({"eval_strategy": "epoch", "eval_steps": 3}, has_eval=True)
    assert strategy == "epoch"
