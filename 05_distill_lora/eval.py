"""学習に入れていない図表を、ベースと SFT で思考オフのまま比べる.

Gemma 4 の vLLM（serve_vllm.sh gemma）へ送る. サンプリングしない.
正解率に加えて、各件の思考と回答を残す.
思考オフでは思考タグが空なので、答えの行より前を思考として残す.

使い方:
    python eval.py --hakusho /path/to/hakushobench
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from hakusho import HakushoSet
from readout import ChartReadout

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class HeldOutCompare:
    """
    学習に残した図表を、ベースと SFT で思考オフ生成し、差分を Markdown にする.

    生成は Gemma 4 の vLLM. サンプリングしない. 正解は「答え:」の右側だけを見る.

    Attributes:
        hakusho (HakushoSet): 図表を PIL にする読み手.
        model_name (str): vLLM に載っているベースモデル名.
        sft_name (str): SFT LoRA の名前.
        sft_dir (Path): `sft.py` が保存した LoRA.
        output_dir (Path): before.jsonl、after.jsonl、heldout.md の保存先.
        max_new_tokens (int): 生成トークンの上限.
        vllm_url (str): Gemma 4 用 vLLM の OpenAI 互換 URL.
        readout (ChartReadout): プロンプトと正解判定の規則.
        resume (bool): True なら既にある original_id を飛ばす.

    Methods:
        reply: 1 件を思考オフで生成し、思考と回答に分ける.
        write: テスト用の図表を、指定したモデルの jsonl に書く.
        write_sides: ベースと SFT を同時に jsonl に書く.
        report: ベースと SFT の Markdown を書く.

    Example:
        compare = HeldOutCompare(
            hakusho=HakushoSet(repo_dir="./hakushobench"),
            output_dir="./outputs/heldout",
        )
        compare.write(test_dataset, compare.output_dir / "before.jsonl", compare.model_name)

    """

    def __init__(
        self,
        hakusho: HakushoSet,
        model_name: str = "google/gemma-4-31B-it",
        sft_name: str = "sft",
        sft_dir: str | Path = "outputs/gemma4_hakusho_sft",
        output_dir: str | Path = "outputs/heldout",
        max_new_tokens: int = 1024,
        vllm_url: str = "http://127.0.0.1:8000",
        resume: bool = False,
    ):
        self.hakusho = hakusho
        self.model_name = model_name
        self.sft_name = sft_name
        self.sft_dir = Path(sft_dir)
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.vllm_url = vllm_url
        self.readout = ChartReadout()
        self.resume = resume

    def reply(self, row: dict, model_id: str) -> dict:
        """
        1 枚の図表を思考オフで読ませ、思考・読み取り・回答と正解かどうかを返す.

        思考タグが空のときは、答えの行より前を思考の欄に回す.
        正解は「答え:」の右側だけを見る.

        Args:
            row (dict): `question`、`answer`、`original_id`、`image` を持つ 1 行.
            model_id (str): ベースなら `model_name`. LoRA なら `sft_name`.

        Returns:
            dict: 識別子、質問、正解、生成全文、思考、読み取り、回答、`hit`.

        Example:
            record = compare.reply(test_dataset[0], compare.sft_name)

        """
        question = str(row["question"]).strip()
        gold = str(row["answer"]).strip()
        raw = self._vllm_reply(self.hakusho.load_chart(row), question, model_id)
        think, grounds, answer_line = self._parts(raw)
        body = self.readout.answer_body(raw)
        return {
            "original_id": str(row["original_id"]),
            "category": str(row.get("category", "")),
            "image_type": str(row.get("image_type", "")),
            "question": question,
            "answer": gold,
            "raw": raw,
            "think": think,
            "grounds": grounds,
            "answer_line": answer_line,
            "hit": body is not None and body == self.readout.norm(gold),
        }

    def reachable(self) -> bool:
        """
        vLLM の /v1/models に届くかを返す.

        Returns:
            bool: 200 番台なら True.

        Example:
            if compare.reachable():
                print("vllm")

        """
        request = urllib.request.Request(
            self.vllm_url.rstrip("/") + "/v1/models",
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return 200 <= response.status < 300
        except urllib.error.URLError:
            return False

    def served_ids(self) -> set[str]:
        """
        vLLM に載っているモデル名を返す.

        Returns:
            set[str]: `/v1/models` の id.

        Example:
            names = compare.served_ids()

        """
        request = urllib.request.Request(
            self.vllm_url.rstrip("/") + "/v1/models",
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"vLLM に接続できません: {self.vllm_url} ({error})"
            ) from error
        return {str(item["id"]) for item in body.get("data", [])}

    def require_lora(
        self,
        label: str,
        adapter_dir: Path,
        lora_name: str,
        served: set[str],
    ) -> None:
        """
        アダプタファイルと vLLM 上の名前が揃っているかを確認する.

        Args:
            label (str): エラー文に出す「SFT」.
            adapter_dir (Path): adapter_config.json があるディレクトリ.
            lora_name (str): vLLM に載っているはずの名前.
            served (set[str]): `/v1/models` の id.

        Returns:
            None: 揃っていれば何もしない. 無ければプロセスを終了する.

        Example:
            compare.require_lora("SFT", compare.sft_dir, compare.sft_name, served)

        """
        if not (adapter_dir / "adapter_config.json").is_file():
            raise SystemExit(
                f"{label} の LoRA がありません: {adapter_dir}\n"
                "adapter がある状態で bash serve_vllm.sh gemma を起動し直してください."
            )
        if lora_name not in served:
            raise SystemExit(
                f"vLLM に LoRA `{lora_name}` が載っていません.\n"
                "adapter がある状態で bash serve_vllm.sh gemma を起動し直してください."
            )

    def _vllm_reply(self, chart, question: str, model_id: str) -> str:
        """
        図表と質問を vLLM に送り、本文を返す.

        Args:
            chart (PIL.Image.Image): RGB の図表.
            question (str): 問い.
            model_id (str): リクエストの `model`.

        Returns:
            str: 生成本文.

        Example:
            raw = compare._vllm_reply(chart, question, "sft")

        """
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._png_data_url(chart)},
                        },
                        {"type": "text", "text": self.readout.prompt(question)},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "max_soft_tokens": 1120,
            },
        }
        request = urllib.request.Request(
            self.vllm_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"vLLM に接続できません: {self.vllm_url} ({error})"
            ) from error
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning and "<think>" not in content:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        return content

    def _png_data_url(self, chart) -> str:
        """
        図表を PNG の data URL にして返す.

        Args:
            chart (PIL.Image.Image): RGB の図表.

        Returns:
            str: `data:image/png;base64,...` の URL.

        Example:
            url = compare._png_data_url(chart)

        """
        buffer = BytesIO()
        chart.save(buffer, format="PNG")
        encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _parts(self, raw: str) -> tuple[str, str, str]:
        """
        生成文を、思考タグの中身、答えより前の行、答えの行に分ける.

        Args:
            raw (str): デコードした生成. 特殊トークンを含み得る.

        Returns:
            tuple[str, str, str]: 思考タグの中身、答えより前の行、答えの行. 無ければ空文字.

        Example:
            think, grounds, answer_line = compare._parts("項目\\n答え: 3.4倍\\n")

        """
        start = raw.find("<think>")
        end = raw.find("</think>")
        if start >= 0 and end > start:
            think = raw[start + len("<think>") : end].strip()
        else:
            think = ""
        lines = self.readout.visible_lines(raw)
        index = self.readout._answer_index(lines)
        if index is None:
            return think, "\n".join(lines), ""
        return think, "\n".join(lines[:index]), lines[index]

    def write(self, test_dataset, path: Path, model_id: str) -> None:
        """
        テスト用の図表を順に生成し、jsonl に書く.

        既定ではその jsonl を作り直す. resume のときだけ既にある original_id を飛ばす.
        空の質問や正解も飛ばす.

        Args:
            test_dataset (object): `split_dataset` のテスト用図表.
            path (Path): 追記する jsonl.
            model_id (str): ベースなら `model_name`. LoRA なら `sft_name`.

        Returns:
            None: jsonl に 1 件ずつ書く.

        Example:
            compare.write(test_dataset, Path("before.jsonl"), compare.model_name)

        """
        done = self._ready_jsonl(path)
        hits = 0
        seen = 0
        with path.open("a", encoding="utf-8") as traces:
            for index, row in enumerate(test_dataset, start=1):
                question = str(row["question"]).strip()
                gold = str(row["answer"]).strip()
                original_id = str(row["original_id"])
                if not question or not gold or original_id in done:
                    continue
                record = self.reply(row, model_id)
                traces.write(json.dumps(record, ensure_ascii=False) + "\n")
                traces.flush()
                seen += 1
                hits += int(record["hit"])
                print(
                    f"{path.name} {index}/{len(test_dataset)} hit {hits}/{seen} {original_id}",
                    flush=True,
                )

    def write_sides(self, test_dataset, sides: list[tuple[Path, str]]) -> None:
        """
        テスト用の図表を、指定したモデルへ同時に送り、それぞれの jsonl に書く.

        既定では jsonl を作り直す. resume のときだけ、既にある original_id はその側だけ飛ばす.

        Args:
            test_dataset (object): `split_dataset` のテスト用図表.
            sides (list[tuple[Path, str]]): (jsonl のパス, vLLM の model id) の列.

        Returns:
            None: 各 jsonl に書く.

        Example:
            compare.write_sides(
                test_dataset,
                [
                    (Path("before.jsonl"), compare.model_name),
                    (Path("after.jsonl"), compare.sft_name),
                ],
            )

        """
        done_ids = [self._ready_jsonl(path) for path, _model_id in sides]
        hits = [0] * len(sides)
        seens = [0] * len(sides)
        traces = [path.open("a", encoding="utf-8") for path, _model_id in sides]
        try:
            for index, row in enumerate(test_dataset, start=1):
                question = str(row["question"]).strip()
                gold = str(row["answer"]).strip()
                original_id = str(row["original_id"])
                if not question or not gold:
                    continue
                need = [
                    original_id not in done_ids[side_index]
                    for side_index in range(len(sides))
                ]
                if not any(need):
                    continue
                with ThreadPoolExecutor(max_workers=len(sides)) as pool:
                    futures = [
                        pool.submit(self.reply, row, sides[side_index][1])
                        if need[side_index]
                        else None
                        for side_index in range(len(sides))
                    ]
                    for side_index, future in enumerate(futures):
                        if future is None:
                            continue
                        record = future.result()
                        traces[side_index].write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        traces[side_index].flush()
                        seens[side_index] += 1
                        hits[side_index] += int(record["hit"])
                counts = " ".join(
                    f"{sides[side_index][0].stem} "
                    f"{hits[side_index]}/{seens[side_index]}"
                    for side_index in range(len(sides))
                )
                print(f"{index}/{len(test_dataset)} {counts} {original_id}", flush=True)
        finally:
            for handle in traces:
                handle.close()

    def _ready_jsonl(self, path: Path) -> set[str]:
        """
        jsonl の親ディレクトリを作り、resume でなければファイルを消して空にする.

        Args:
            path (Path): 書き込む jsonl.

        Returns:
            set[str]: 飛ばす original_id. 作り直すときは空.

        Example:
            done = compare._ready_jsonl(Path("before.jsonl"))

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.resume and path.is_file():
            path.unlink()
            return set()
        return self._done_ids(path)

    def _done_ids(self, path: Path) -> set[str]:
        """
        jsonl に既にある original_id を返す.

        Args:
            path (Path): 読み込む jsonl.

        Returns:
            set[str]: 生成を飛ばす id. ファイルが無ければ空.

        Example:
            done = compare._done_ids(Path("before.jsonl"))

        """
        if not path.is_file():
            return set()
        done = set()
        with path.open(encoding="utf-8") as traces:
            for line in traces:
                if line.strip():
                    done.add(str(json.loads(line)["original_id"]))
        return done

    def report(self, before_path: Path, after_path: Path, report_path: Path) -> None:
        """
        ベースと SFT の jsonl から、正解率と各件の思考・回答を Markdown にする.

        無い側は未生成として書く.

        Args:
            before_path (Path): ベースモデルの jsonl.
            after_path (Path): SFT の jsonl.
            report_path (Path): 書き出す Markdown.

        Returns:
            None: Markdown を report_path に書く.

        Example:
            compare.report(before_path, after_path, Path("heldout.md"))

        """
        before = self._load_jsonl(before_path)
        after = self._load_jsonl(after_path)
        order = list(before) or list(after)
        for original_id in after:
            if original_id not in order:
                order.append(original_id)
        lines = self._summary(before, after, order)
        for index, original_id in enumerate(order, start=1):
            lines.append(
                self._item(index, before.get(original_id), after.get(original_id))
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("report", report_path.resolve())

    def _load_jsonl(self, path: Path) -> dict[str, dict]:
        """
        jsonl を original_id からレコードへの辞書にして返す.

        Args:
            path (Path): 読み込む jsonl.

        Returns:
            dict[str, dict]: ファイルが無ければ空.

        Example:
            rows = compare._load_jsonl(Path("before.jsonl"))

        """
        if not path.is_file():
            return {}
        rows = {}
        with path.open(encoding="utf-8") as traces:
            for line in traces:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows[str(record["original_id"])] = record
        return rows

    def _summary(self, before: dict, after: dict, order: list[str]) -> list[str]:
        """
        正解率と、正解が入れ替わった件の一覧を Markdown の先頭にする.

        Args:
            before (dict): ベースのレコード.
            after (dict): SFT のレコード.
            order (list[str]): 見出しの順に使う original_id.

        Returns:
            list[str]: 見出しと集計の行.

        Example:
            head = compare._summary(before, after, order)

        """
        base_sft = [
            original_id
            for original_id in order
            if original_id in before and original_id in after
        ]
        lines = [
            "# 学習に入れていない図表",
            "",
            "思考はオフ. 正解は「答え:」の右側がデータセットの正解と一致した件数.",
            "",
            self._hit_line("ベース", before),
            self._hit_line("SFT", after),
            f"- ベースと SFT がある件数: {len(base_sft)}",
            "",
        ]
        lines.extend(
            self._flip_lines(
                "ベース不正解 → SFT正解",
                "ベース正解 → SFT不正解",
                before,
                after,
                base_sft,
            )
        )
        return lines

    def _hit_line(self, label: str, rows: dict) -> str:
        """
        1 側の正解件数を Markdown の箇条書き 1 行にする.

        Args:
            label (str): 「ベース」または「SFT」.
            rows (dict): その側のレコード. 空なら未生成.

        Returns:
            str: `- ラベル: 正解/件数` または未生成.

        Example:
            line = compare._hit_line("SFT", after)

        """
        if not rows:
            return f"- {label}: 未生成"
        hits = sum(int(rows[original_id]["hit"]) for original_id in rows)
        return f"- {label}: {hits}/{len(rows)}"

    def _flip_lines(
        self,
        gained_title: str,
        lost_title: str,
        older: dict,
        newer: dict,
        ids: list[str],
    ) -> list[str]:
        """
        2 側のあいだで正解が入れ替わった id を見出し付きの箇条書きにする.

        Args:
            gained_title (str): 不正解から正解になった件の見出し.
            lost_title (str): 正解から不正解になった件の見出し.
            older (dict): 比較の左側.
            newer (dict): 比較の右側.
            ids (list[str]): 両方にある original_id.

        Returns:
            list[str]: 見出しと id の行. 入れ替わりが無ければ空.

        Example:
            lines = compare._flip_lines(
                "ベース不正解 → SFT正解",
                "ベース正解 → SFT不正解",
                before,
                after,
                both_ids,
            )

        """
        gained = [
            original_id
            for original_id in ids
            if not older[original_id]["hit"] and newer[original_id]["hit"]
        ]
        lost = [
            original_id
            for original_id in ids
            if older[original_id]["hit"] and not newer[original_id]["hit"]
        ]
        lines = []
        if gained:
            lines.append(f"## {gained_title}")
            lines.append("")
            for original_id in gained:
                lines.append(f"- {original_id}: {older[original_id]['question']}")
            lines.append("")
        if lost:
            lines.append(f"## {lost_title}")
            lines.append("")
            for original_id in lost:
                lines.append(f"- {original_id}: {older[original_id]['question']}")
            lines.append("")
        return lines

    def _item(self, index: int, before: dict | None, after: dict | None) -> str:
        """
        1 件の質問、正解、ベースと SFT の思考と回答を Markdown にする.

        Args:
            index (int): 見出しの番号. 1 始まり.
            before (dict | None): ベースのレコード. 未生成なら None.
            after (dict | None): SFT のレコード. 未生成なら None.

        Returns:
            str: 1 件分の Markdown.

        Example:
            section = compare._item(1, before_row, after_row)

        """
        sample = before or after
        head = (
            f"## {index}. {sample['original_id']} "
            f"{sample['image_type']} {sample['category']}"
        )
        sections = [
            head,
            "",
            "質問",
            "",
            sample["question"],
            "",
            "正解",
            "",
            sample["answer"],
            "",
            self._side("ベース", before),
            self._side("SFT", after),
        ]
        return "\n".join(sections)

    def _side(self, label: str, record: dict | None) -> str:
        """
        片側の思考、読み取り、回答、生成全文を Markdown にする.

        Args:
            label (str): 「ベース」または「SFT」.
            record (dict | None): その側の生成. 未生成なら None.

        Returns:
            str: 見出しから生成全文まで.

        Example:
            block = compare._side("ベース", record)

        """
        if record is None:
            return f"### {label}\n\n未生成\n"
        mark = "正解" if record["hit"] else "不正解"
        think = record["think"].strip()
        grounds = record["grounds"].strip()
        if think:
            thought = think
        else:
            thought = grounds or "（思考タグは空で、答えの前の行もない）"
        answer = record["answer_line"].strip() or "（答えの行なし）"
        return "\n".join(
            [
                f"### {label}: {mark}",
                "",
                "思考",
                "",
                self._fence(thought),
                "",
                "回答",
                "",
                self._fence(answer),
                "",
                "生成全文",
                "",
                self._fence(record["raw"]),
                "",
            ]
        )

    def _fence(self, text: str) -> str:
        """
        生成文を、Markdown のコードフェンスに入れて返す.

        Args:
            text (str): そのまま見せる文.

        Returns:
            str: フェンスで囲んだ文. 中のフェンスは引用符に替える.

        Example:
            fenced = compare._fence("答え: 3.4倍")

        """
        body = text.replace("```", "'''")
        return f"```\n{body}\n```"


def main():
    parser = argparse.ArgumentParser(
        description="学習に入れていない図表を、ベースと SFT で思考オフのまま比べる."
    )
    parser.add_argument("--hakusho", required=True, help="clone した hakushobench のパス.")
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument(
        "--sft-name",
        default="sft",
        help="serve_vllm.sh gemma の --lora-modules に付けた SFT の名前.",
    )
    parser.add_argument(
        "--sft-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "gemma4_hakusho_sft"),
        help="sft.py が保存した LoRA.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "heldout"),
        help="jsonl と Markdown の保存先.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--side",
        choices=("before", "after", "sft", "both", "report"),
        default="both",
        help="before はベース、sft/after は SFT、both は両方、report は Markdown だけ.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=HakushoSet.EVAL_RATIO,
        help="テストに回す割合. distill_data.py と同じ値にする.",
    )
    parser.add_argument("--seed", type=int, default=HakushoSet.SEED)
    parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
        help="Gemma 4 用 vLLM の待ち受け URL.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既にある original_id は飛ばす. 省略時は jsonl を作り直す.",
    )
    args = parser.parse_args()
    side = {"after": "sft"}.get(args.side, args.side)

    output_dir = Path(args.output_dir)
    before_path = output_dir / "before.jsonl"
    after_path = output_dir / "after.jsonl"
    report_path = output_dir / "heldout.md"
    compare = HeldOutCompare(
        hakusho=HakushoSet(repo_dir=args.hakusho),
        model_name=args.model,
        sft_name=args.sft_name,
        sft_dir=args.sft_dir,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        vllm_url=args.vllm_url,
        resume=args.resume,
    )
    if side == "report":
        compare.report(before_path, after_path, report_path)
        return
    if not compare.reachable():
        raise SystemExit(
            f"vLLM がありません: {args.vllm_url}\n"
            "先に bash serve_vllm.sh gemma を起動してください."
        )
    served = compare.served_ids()
    print("vllm", ", ".join(sorted(served)))
    if side in ("sft", "both"):
        compare.require_lora("SFT", compare.sft_dir, compare.sft_name, served)

    hakusho = compare.hakusho
    rows = hakusho.filter_charts(hakusho.load_rows())
    train_dataset, test_dataset = hakusho.split_dataset(
        rows,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print("train", len(train_dataset), "test", len(test_dataset))
    sides = []
    if side in ("before", "both"):
        sides.append((before_path, compare.model_name))
    if side in ("sft", "both"):
        sides.append((after_path, compare.sft_name))
    compare.write_sides(test_dataset, sides)
    compare.report(before_path, after_path, report_path)


if __name__ == "__main__":
    main()
