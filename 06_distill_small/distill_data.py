"""蒸留・自己生成用の教師データを作る. 思考オンで書かせ、正解と一致した思考＋根拠＋答えを残す.

Step 1（生徒の自己生成）: `bash serve_vllm.sh gemma` を起動して実行すると、Gemma 4 E4B（思考オン）で解き、自力で正解した思考＋回答を保存する.
Step 2（教師で補完）: `bash serve_vllm.sh qwen` に切り替えて同じコマンドを実行すると、Step 1 で不正解だった問題だけを Qwen3.8-27B（思考オン）で解き、正解した思考＋回答で補完する.

使い方:
    python distill_data.py --hakusho /path/to/hakushobench
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from hakusho import HakushoSet
from readout import ChartReadout


class ThinkAnswerStop:
    """
    思考を閉じたあとの「答え:」の行が改行まで出た生成を止める.

    思考の途中に「答え:」と出ても止めない. 改行トークンのときだけ文字列を見る.

    Attributes:
        tokenizer (object): 生成トークンを文字列に戻す processor または tokenizer.
        prompt_len (int): 左パディングを含めたプロンプト長.
        newline_ids (set[int]): 改行トークンの id.
        readout (ChartReadout): 答えの行が閉じたか見る規則.
        torch (object): 停止フラグのテンソルを作る torch.

    Methods:
        __call__: バッチの各行を止めるかを返す.

    Example:
        stop = ThinkAnswerStop(
            tokenizer,
            prompt_len=128,
            newline_ids={198},
            readout=ChartReadout(),
            torch=torch,
        )

    """

    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        newline_ids: set[int],
        readout: ChartReadout,
        torch,
    ):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.newline_ids = newline_ids
        self.readout = readout
        self.torch = torch

    def __call__(self, input_ids, scores, **kwargs):
        """
        思考のあとに答えの行が閉じていれば、その行を True にする.

        Args:
            input_ids (object): プロンプトと、そこまでの生成.
            scores (object): 未使用の次トークンスコア.
            **kwargs: `generate` が渡す未使用の引数.

        Returns:
            object: 行ごとの停止フラグ. バッチと同じ長さ.

        Example:
            flags = stop(input_ids, scores)

        """
        del scores, kwargs
        flags = []
        for row in input_ids:
            if int(row[-1]) not in self.newline_ids:
                flags.append(False)
                continue
            decoded_text = self.tokenizer.decode(
                row[self.prompt_len :].tolist(),
                skip_special_tokens=False,
            )
            flags.append(self.readout.answer_closed(decoded_text))
        return self.torch.tensor(flags, device=input_ids.device)


class DistillSet:
    """
    Gemma 4 E4B または Qwen3.8-27B を思考オンで動かし、正解と一致した思考と回答を jsonl に残す.

    すでに `kept: True`（思考あり）の original_id は再生成しない.
    別モデルで `kept: False` だった original_id は、現在のモデルで再挑戦して上書きする.

    Attributes:
        SOFT_TOKENS (int): Gemma 4 に渡す画像ソフトトークン上限. 1120.
        hakusho (HakushoSet): 図表を PIL にする読み手.
        model_name (str): Hugging Face 上のモデル名（`auto` なら vLLM から自動取得）.
        max_new_tokens (int): 思考オンの生成上限.
        batch_size (int): 同時に扱う図表の枚数.
        backend (str): `vllm` か `transformers`.
        vllm_url (str): vLLM の OpenAI 互換 URL.
        traces_path (Path): 残した文と、捨てた理由の jsonl.
        readout (ChartReadout): プロンプトと採否の規則.
        model (object | None): transformers のときだけ持つモデル.
        processor (object | None): transformers のときだけ持つ processor.

    Methods:
        reachable: vLLM の /v1/models に届くかを返す.
        detect_vllm_model: vLLM に載っているベースモデル名を返す.
        load: transformers で 16bit のベースモデルを読む.
        records: 複数枚を生成し、残すか捨てるかの辞書を返す.
        write: 学習用の図表を順に生成し、jsonl を更新する.

    Example:
        distill = DistillSet(
            hakusho=HakushoSet(repo_dir="./hakushobench"),
            traces_path="./outputs/qwen38_hakusho_sft/traces.jsonl",
        )
        distill.write(rows)

    """

    SOFT_TOKENS = 1120

    def __init__(
        self,
        hakusho: HakushoSet,
        model_name: str = "auto",
        max_new_tokens: int = 2048,
        batch_size: int = 16,
        backend: str = "vllm",
        vllm_url: str = "http://127.0.0.1:8000",
        traces_path: str | Path = "outputs/qwen38_hakusho_sft/traces.jsonl",
    ):
        self.hakusho = hakusho
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.backend = backend
        self.vllm_url = vllm_url
        self.traces_path = Path(traces_path)
        self.readout = ChartReadout()
        self.model = None
        self.processor = None
        self._torch = None

    def reachable(self) -> bool:
        """
        vLLM のモデル一覧に届くかを返す.

        Returns:
            bool: 200 番台なら True. 接続できなければ False.

        Example:
            if distill.reachable():
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

    def detect_vllm_model(self) -> str:
        """
        vLLM の `/v1/models` からベースモデル名を取得して返す.

        LoRA 名（`sft` など）が混ざっていても、ベースモデル側を優先して返す.

        Returns:
            str: vLLM に載っているベースモデル名.

        Example:
            model_name = distill.detect_vllm_model()

        """
        request = urllib.request.Request(
            self.vllm_url.rstrip("/") + "/v1/models",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
        model_ids = [str(item["id"]) for item in body.get("data", [])]
        for model_id in model_ids:
            if "/" in model_id:
                return model_id
        if model_ids:
            return model_ids[0]
        raise RuntimeError(f"vLLM にモデルが載っていません: {self.vllm_url}")

    def load(self) -> None:
        """
        LoRA を付けず、16bit のベースモデルを transformers で推論用に読む.

        vLLM が無いときだけ呼ぶ.

        Returns:
            None: `model` と `processor` を属性に持たせる.

        Example:
            distill.load()

        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

        if self.model_name == "auto":
            self.model_name = "Qwen/Qwen3.8-27B"
        processor = AutoProcessor.from_pretrained(self.model_name)
        loader = (
            AutoModelForMultimodalLM
            if "gemma" in self.model_name.lower()
            else AutoModelForImageTextToText
        )
        try:
            model = loader.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except TypeError:
            model = loader.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        model.eval()
        self._torch = torch
        self.model = model
        self.processor = processor

    def records(self, batch_rows: list[dict]) -> list[dict]:
        """
        複数枚の図表を思考オンで読ませ、答えが一致した思考＋本文を辞書にする.

        vLLM なら並列 HTTP. transformers なら同じプロセスで generate する.

        Args:
            batch_rows (list[dict]): `question`、`answer`、`original_id`、`image` を持つ行.

        Returns:
            list[dict]: 入力と同じ順. 各要素は `original_id`、`kept`、`reason`、`model`.
            残すときは `question`、`answer`、`thinking`、`completion` も持つ.

        Example:
            traced_rows = distill.records(rows[:4])

        """
        if self.backend == "vllm":
            with ThreadPoolExecutor(max_workers=len(batch_rows)) as pool:
                return list(pool.map(self._vllm_one, batch_rows))
        return self._hf_records(batch_rows)

    def _hf_records(self, batch_rows: list[dict]) -> list[dict]:
        """
        transformers で複数枚をまとめて思考オン生成し、採否の辞書にする.

        Args:
            batch_rows (list[dict]): `question`、`answer`、`original_id`、`image` を持つ行.

        Returns:
            list[dict]: `records` と同じ形.

        Example:
            traced_rows = distill._hf_records(rows[:4])

        """
        torch = self._torch
        processor = self.processor
        texts = []
        charts = []
        is_gemma = "gemma" in self.model_name.lower()
        for row in batch_rows:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": self.readout.prompt(str(row["question"]).strip()),
                        },
                    ],
                }
            ]
            template_kw = {
                "add_generation_prompt": True,
                "tokenize": False,
                "enable_thinking": True,
            }
            if is_gemma:
                template_kw["max_soft_tokens"] = self.SOFT_TOKENS
            else:
                template_kw["reasoning_effort"] = "low"
            try:
                rendered_prompt = processor.apply_chat_template(messages, **template_kw)
            except TypeError:
                template_kw.pop("max_soft_tokens", None)
                template_kw.pop("reasoning_effort", None)
                rendered_prompt = processor.apply_chat_template(messages, **template_kw)
            texts.append(rendered_prompt)
            charts.append(self.hakusho.load_chart(row))
        pad_owner = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        padding_side = getattr(pad_owner, "padding_side", "right")
        pad_owner.padding_side = "left"
        try:
            inputs = processor(
                charts,
                texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            inputs = processor(
                text=texts,
                images=charts,
                padding=True,
                return_tensors="pt",
            )
        pad_owner.padding_side = padding_side
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        prompt_len = inputs["input_ids"].shape[1]
        token_source = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        newline_ids = set()
        if hasattr(token_source, "encode"):
            newline_ids = {
                int(token_id)
                for token_id in token_source.encode("\n", add_special_tokens=False)
            }
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if newline_ids:
            generate_kwargs["stopping_criteria"] = [
                ThinkAnswerStop(
                    processor,
                    prompt_len,
                    newline_ids,
                    self.readout,
                    torch,
                )
            ]
        with torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        decode = (
            processor.decode if hasattr(processor, "decode") else token_source.decode
        )
        traced_rows = []
        for row, row_ids in zip(batch_rows, output_ids):
            raw = decode(row_ids[prompt_len:], skip_special_tokens=False)
            traced_rows.append(self._trace(row, raw))
        return traced_rows

    def _vllm_one(self, row: dict) -> dict:
        """
        1 枚を vLLM に送って採否の辞書にする.

        Args:
            row (dict): `question`、`answer`、`original_id`、`image` を持つ 1 行.

        Returns:
            dict: `_trace` と同じ形.

        Example:
            traced = distill._vllm_one(rows[0])

        """
        question = str(row["question"]).strip()
        chart = self.hakusho.load_chart(row)
        buffer = BytesIO()
        chart.save(buffer, format="PNG")
        encoded_png = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        image_data_url = f"data:image/png;base64,{encoded_png}"
        template_kwargs = {"enable_thinking": True}
        if "gemma" in self.model_name.lower():
            template_kwargs["max_soft_tokens"] = self.SOFT_TOKENS
        else:
            template_kwargs["reasoning_effort"] = "low"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {"type": "text", "text": self.readout.prompt(question)},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": template_kwargs,
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
        if reasoning and "<think>" not in content and "<|channel>thought" not in content:
            raw = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            raw = content
        return self._trace(row, raw)

    def _trace(self, row: dict, raw: str) -> dict:
        """
        生成文を採点し、思考と本文を含む辞書にする.

        Args:
            row (dict): `question`、`answer`、`original_id` を持つ 1 行.
            raw (str): モデル出力. 思考チャネルまたは思考タグを含む.

        Returns:
            dict: `original_id`、`kept`、`reason`、`model`.
            残すときは `question`、`answer`、`thinking`、`completion` も持つ.

        Example:
            traced = distill._trace(rows[0], "<think>確認</think>\\n答え: 3.4倍\\n")

        """
        question = str(row["question"]).strip()
        gold = str(row["answer"]).strip()
        completion, reason = self.readout.judge(raw, gold, question=question)
        traced = {
            "original_id": str(row["original_id"]),
            "kept": reason == "kept",
            "reason": reason,
            "model": self.model_name,
        }
        if completion is None:
            return traced
        traced["question"] = question
        traced["answer"] = gold
        traced["thinking"] = self.readout.extract_thought(raw) or self.readout.grounds(raw).strip()
        traced["completion"] = completion
        return traced

    def write(self, rows) -> Counter:
        """
        学習用の図表を batch_size 枚ずつ生成し、jsonl を更新する.

        すでに `kept: True`（思考あり）の original_id は飛ばす.
        同じモデルで `kept: False` だった original_id も飛ばす.
        別モデル（例: Gemma 4 のあとの Qwen3.8）では、`kept: False` だった問題だけ再生成して上書きする.

        Args:
            rows (object): `original_id` を持つ学習用の図表.

        Returns:
            Counter: この実行で処理した理由ごとの件数.

        Example:
            counts = distill.write(rows)

        """
        existing_records = {}
        if self.traces_path.is_file():
            with self.traces_path.open(encoding="utf-8") as traces:
                for line in traces:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    existing_records[str(record["original_id"])] = record

        counts = Counter()
        pending = []
        pending_index = []
        total = len(rows)
        self.traces_path.parent.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows, start=1):
            question = str(row["question"]).strip()
            gold = str(row["answer"]).strip()
            original_id = str(row["original_id"])
            if not question or not gold:
                continue
            prev = existing_records.get(original_id)
            if prev is not None:
                if prev.get("kept") and prev.get("thinking"):
                    continue
                if prev.get("model") == self.model_name and prev.get("thinking") is not None:
                    continue
            pending.append(row)
            pending_index.append(index)
            if len(pending) < self.batch_size:
                continue
            self._write_batch(pending, pending_index, existing_records, counts, total)
            pending = []
            pending_index = []
        if pending:
            self._write_batch(pending, pending_index, existing_records, counts, total)
        return counts

    def _write_batch(
        self,
        batch_rows: list[dict],
        indexes: list[int],
        existing_records: dict[str, dict],
        counts: Counter,
        total: int,
    ) -> None:
        """
        1 バッチを生成し、辞書と jsonl ファイルと進捗表示を更新する.

        Args:
            batch_rows (list[dict]): このバッチの図表.
            indexes (list[int]): 各図表の 1 始まりの番号.
            existing_records (dict[str, dict]): 全 original_id の最新レコード辞書.
            counts (Counter): 理由ごとの件数.
            total (int): 学習用の図表の総数.

        Returns:
            None: jsonl を書き直し、進捗を表示する.

        Example:
            distill._write_batch(pending, indexes, existing_records, counts, 912)

        """
        traced_rows = self.records(batch_rows)
        for index, traced in zip(indexes, traced_rows):
            existing_records[str(traced["original_id"])] = traced
            counts[traced["reason"]] += 1
            total_kept = sum(
                1
                for item in existing_records.values()
                if item.get("kept") and item.get("thinking")
            )
            print(
                f"[{self.model_name}] {index}/{total} {traced['reason']} "
                f"batch_kept {counts['kept']} total_kept {total_kept} seen {sum(counts.values())}",
                flush=True,
            )
        with self.traces_path.open("w", encoding="utf-8") as handle:
            for item in existing_records.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="蒸留・自己生成用の教師データを作る. 思考オンで書かせ、正解と一致した思考と本文を残す."
    )
    parser.add_argument(
        "--hakusho", required=True, help="clone した hakushobench のパス."
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="モデル名. auto なら起動中の vLLM（gemma または qwen）から自動検出する.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="同時に扱う枚数. 省略時は vLLM なら 16、transformers なら 4.",
    )
    parser.add_argument(
        "--traces",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "qwen38_hakusho_sft"
            / "traces.jsonl"
        ),
        help="残した思考＋本文と捨てた理由の jsonl.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=HakushoSet.EVAL_RATIO,
        help="テストに回す割合. sft.py と同じ値にする.",
    )
    parser.add_argument("--seed", type=int, default=HakushoSet.SEED)
    parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
        help="vLLM（serve_vllm.sh gemma または qwen）の待ち受け URL.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "transformers"),
        default="auto",
        help="auto は vLLM が届けばそれを使い、ダメなら transformers.",
    )
    args = parser.parse_args()

    hakusho = HakushoSet(repo_dir=args.hakusho)
    rows = hakusho.filter_charts(hakusho.load_rows())
    train_dataset, test_dataset = hakusho.split_dataset(
        rows,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print("train", len(train_dataset), "test", len(test_dataset))
    distill = DistillSet(
        hakusho=hakusho,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        vllm_url=args.vllm_url,
        traces_path=args.traces,
    )
    vllm_up = distill.reachable()
    if args.backend == "transformers" or (args.backend == "auto" and not vllm_up):
        backend = "transformers"
    elif args.backend == "vllm" and not vllm_up:
        print("vLLM に届かないので transformers で生成します:", args.vllm_url)
        backend = "transformers"
    else:
        backend = "vllm"
    if backend == "vllm" and distill.model_name == "auto":
        distill.model_name = distill.detect_vllm_model()
    if args.batch_size is None:
        batch_size = 16 if backend == "vllm" else 4
    else:
        batch_size = args.batch_size
    distill.backend = backend
    distill.batch_size = batch_size
    print("backend", backend, "model", distill.model_name, "batch_size", batch_size)
    if backend == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        distill.load()
    counts = distill.write(train_dataset)
    print("wrote", distill.traces_path.resolve())
    print(
        "kept",
        counts["kept"],
        "no_thought",
        counts["no_thought"],
        "english_thought",
        counts["english_thought"],
        "no_answer",
        counts["no_answer"],
        "mismatch",
        counts["mismatch"],
        "too_long",
        counts["too_long"],
        "other_as_country",
        counts["other_as_country"],
    )


if __name__ == "__main__":
    main()
