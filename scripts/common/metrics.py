from __future__ import annotations


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def length_ratio(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    return min(len(pred), len(gold)) / max(len(pred), len(gold), 1)


def _tokens(text: str) -> list[str]:
    return [part for part in text.strip().lower().replace("\n", " ").split(" ") if part]


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = 0
    remaining = list(gold_tokens)
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_len(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for j, other in enumerate(right, start=1):
            if item == other:
                current.append(prev[j - 1] + 1)
            else:
                current.append(max(current[-1], prev[j]))
        prev = current
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(gold, pred)["rougeL"].fmeasure)
    except Exception:
        pred_tokens = _tokens(pred)
        gold_tokens = _tokens(gold)
        if not pred_tokens or not gold_tokens:
            return 0.0
        lcs = lcs_len(pred_tokens, gold_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def bleu(pred: str, gold: str) -> float:
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu([pred], [[gold]]).score) / 100.0
    except Exception:
        return token_f1(pred, gold)


def assistant_turns(messages: list[dict[str, str]]) -> list[tuple[list[dict[str, str]], str]]:
    """Split a multi-turn sample into (prompt_messages, gold_assistant) pairs."""
    pairs: list[tuple[list[dict[str, str]], str]] = []
    prefix: list[dict[str, str]] = []
    for item in messages:
        role = item.get("role")
        if role == "assistant":
            pairs.append((list(prefix), item.get("content") or ""))
        prefix.append(item)
    return pairs
