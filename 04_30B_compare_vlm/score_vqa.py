"""pdf_compare.py が出力したID表を vqa_gt.json と突き合わせて正解率を出す。"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GT_PATH = ROOT / "vqa_gt.json"
ANSWER_FILES = [
    ("Qwen3.8 thinking-on", ROOT / "pdf_Qwen3.8-27B_thinking-on.md"),
    ("Qwen3.8 thinking-off", ROOT / "pdf_Qwen3.8-27B_thinking-off.md"),
    ("Qwen3.6-27B", ROOT / "pdf_Qwen3.6-27B.md"),
    ("Gemma 4 31B", ROOT / "pdf_gemma-4-31B-it.md"),
    ("Muse Glimmer 30B", ROOT / "pdf_Muse-Glimmer-30B.md"),
]
OUT_MD = ROOT / "eval_vqa.md"
QID_TITLES = {"q1": "問1 表のセル", "q2": "問2 軸と注記", "q3": "問3 円グラフ"}

# 比較前に落とす文字。小数点は残す（3.3 と 33 を区別するため）。
_DROP_RE = re.compile(r"[\s,%\"'`*~_「」『』()\[\]【】。、・:;=\-ー―–—/|]")
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_BLANK_WORDS = {"記載なし", "なし", "不明", "読み取れない", "N/A", "NA"}


@dataclass
class ItemResult:
    item_id: str
    qid: str
    question: str
    gold: str
    raw: str
    correct: bool
    reason: str


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.strip())
    return _DROP_RE.sub("", text)


def as_number(text: str) -> float | None:
    return float(text) if _NUM_RE.match(text) else None


def parse_answers(text: str, valid_ids: set[str]) -> dict[str, str]:
    """Markdown表から ID -> 答えセル を拾う。"""
    found: dict[str, str] = {}
    for line in text.splitlines():
        row = _ROW_RE.match(line)
        if not row:
            continue
        cells = [c.strip() for c in row.group(1).split("|")]
        item_id = normalize(cells[0]).lower()
        if item_id not in valid_ids:
            continue
        if len(cells) >= 3:
            answer = cells[-1]
        elif len(cells) == 2:
            answer = cells[1]
        else:
            answer = ""
        previous = found.get(item_id)
        if previous and normalize(previous) and not normalize(answer):
            continue
        found[item_id] = answer
    return found


def score_item(raw: str, item: dict) -> ItemResult:
    got = normalize(raw)

    def result(correct: bool, reason: str) -> ItemResult:
        return ItemResult(
            item["id"], item["qid"], item["question"], item["gold"], raw, correct, reason
        )

    if not got or got in {normalize(w) for w in _BLANK_WORDS}:
        return result(False, "無回答")

    got_num = as_number(got)
    for gold in [item["gold"], *item.get("accept", [])]:
        gold_norm = normalize(gold)
        if got == gold_norm:
            return result(True, "一致")
        gold_num = as_number(gold_norm)
        if got_num is not None and gold_num is not None and got_num == gold_num:
            return result(True, "数値一致")
    return result(False, "不一致")


def score_model(name: str, path: Path, items: list[dict]) -> dict:
    valid_ids = {item["id"] for item in items}
    answers = parse_answers(path.read_text(encoding="utf-8"), valid_ids)
    results = [score_item(answers.get(item["id"], ""), item) for item in items]
    by_q: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_q.setdefault(r.qid, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(r.correct)
    correct = sum(1 for r in results if r.correct)
    return {
        "name": name,
        "path": path.name,
        "correct": correct,
        "total": len(results),
        "acc": correct / len(results) if results else 0.0,
        "by_q": by_q,
        "results": {r.item_id: r for r in results},
        "misses": [r for r in results if not r.correct],
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def cell(result: ItemResult) -> str:
    shown = " ".join(result.raw.split()) or "（空欄）"
    if len(shown) > 24:
        shown = shown[:23] + "…"
    shown = shown.replace("|", "\\|")
    return shown if result.correct else f"✗ {shown}"


def build_markdown(items: list[dict], scored: list[dict]) -> str:
    lines = [
        "# PDF読み取り正解率",
        "",
        "各モデルに短答のID表を埋めさせ、`vqa_gt.json` の正解と完全一致で採点した。",
        f"項目数: **{len(items)}**（{QID_TITLES['q1']} / {QID_TITLES['q2']} / {QID_TITLES['q3']}）。",
        "",
        "```text",
        "Accuracy = 正解項目数 / 全項目数",
        "```",
        "",
        "## 総合",
        "",
        "| 順位 | モデル | 正解数 | 項目数 | Accuracy |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, s in enumerate(scored, start=1):
        lines.append(
            f"| {i} | {s['name']} | {s['correct']} | {s['total']} | **{pct(s['acc'])}** |"
        )

    lines += ["", "## 問別 Accuracy", ""]
    header = " | ".join(QID_TITLES[q] for q in ("q1", "q2", "q3"))
    lines.append(f"| モデル | {header} |")
    lines.append("|---|---:|---:|---:|")
    for s in scored:
        cells = []
        for qid in ("q1", "q2", "q3"):
            b = s["by_q"].get(qid, {"correct": 0, "total": 0})
            cells.append(
                f"{pct(b['correct'] / b['total'])} ({b['correct']}/{b['total']})"
                if b["total"]
                else "-"
            )
        lines.append(f"| {s['name']} | " + " | ".join(cells) + " |")

    lines += ["", "## 全項目の回答", "", "✗ が付いた欄が不正解。", ""]
    lines.append(
        "| ID | 問い | 正解 | " + " | ".join(s["name"] for s in scored) + " |"
    )
    lines.append("|---|---|---|" + "---|" * len(scored))
    for item in items:
        row = [item["id"], item["question"], item["gold"]]
        row += [cell(s["results"][item["id"]]) for s in scored]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 不正解の内訳", ""]
    for s in scored:
        lines += [f"### {s['name']}", ""]
        if not s["misses"]:
            lines += ["- なし", ""]
            continue
        for m in s["misses"]:
            answer = " ".join(m.raw.split()) or "（空欄）"
            lines.append(
                f"- `{m.item_id}` {m.question} / 正解: {m.gold} / 回答: {answer}（{m.reason}）"
            )
        lines.append("")

    lines += [
        "## 採点ルール",
        "",
        "- 答えセルを正規化（空白・カンマ・%・記号・全角半角を統一）してから完全一致で判定する",
        "- 両辺が数値なら数値として比較するので `3.0` と `3` は同じ扱い",
        "- 固有名詞など表記が揺れる項目は `vqa_gt.json` の `accept` に許容形を列挙している",
        "- 表の行が欠けている、空欄、「記載なし」はいずれも不正解（無回答）",
        "",
        f"正解定義: `{GT_PATH.name}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    items = json.loads(GT_PATH.read_text(encoding="utf-8"))["items"]
    scored = [
        score_model(name, path, items)
        for name, path in ANSWER_FILES
        if path.exists()
    ]
    missing = [path.name for _, path in ANSWER_FILES if not path.exists()]
    for name in missing:
        print(f"スキップ（ファイルなし）: {name}")
    if not scored:
        raise SystemExit("採点対象の pdf_*.md が見つかりません。")
    scored.sort(key=lambda s: (-s["acc"], s["name"]))

    OUT_MD.write_text(build_markdown(items, scored), encoding="utf-8")
    print(OUT_MD)
    for s in scored:
        print(f"{s['name']}: {s['correct']}/{s['total']} = {pct(s['acc'])}")


if __name__ == "__main__":
    main()
