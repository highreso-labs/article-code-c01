"""蒸留用の教師文を作る. 思考オンで書かせ、答えが一致した短い文だけを残す.

学習に使う文は思考を除いた短い型だけ. 一致しない生成は残さない.
学習に使わず残した図表は生成しない. 同じ original_id は jsonl にあればやり直さない.
vLLM が待ち受けていればそこに送る. 届かなければ transformers で生成する. Unsloth は使わない.
待ち受けは `bash serve_vllm.sh qwen` だけ. Gemma 4 のサーバには送らない.

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
        """
        プロンプト長と、改行のときだけ答えの行を見る tokenizer を持たせる.

        Args:
            tokenizer (object): 学習と同じ processor または tokenizer.
            prompt_len (int): 生成前の input_ids の長さ.
            newline_ids (set[int]): 改行として扱うトークン id.
            readout (ChartReadout): 答えの行の判定.
            torch (object): 停止フラグのテンソルを作る torch.

        Returns:
            None: 判定用の属性を持たせる.

        Example:
            stop = ThinkAnswerStop(
                tokenizer,
                prompt_len=128,
                newline_ids={198},
                readout=ChartReadout(),
                torch=torch,
            )

        """
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
            text = self.tokenizer.decode(
                row[self.prompt_len :].tolist(),
                skip_special_tokens=False,
            )
            if "</think>" not in text:
                flags.append(False)
                continue
            visible = text.split("</think>", 1)[1]
            flags.append(self.readout.answer_closed(visible))
        return self.torch.tensor(flags, device=input_ids.device)


class DistillSet:
    """
    Qwen3.8 を思考オンで動かし、正解と一致した短い型だけを jsonl に残す.

    vLLM が生きていれば HTTP で送る. 届かなければ同じモデルを transformers で読む.
    思考の本文は保存しない. すでに jsonl にある original_id は生成しない.

    Attributes:
        hakusho (HakushoSet): 図表を PIL にする読み手.
        model_name (str): Hugging Face 上の Qwen3.8-27B.
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
        load: transformers で 16bit のベースモデルを読む.
        records: 複数枚を生成し、残すか捨てるかの辞書を返す.
        write: 学習用の図表を順に生成し、jsonl に足す.

    Example:
        distill = DistillSet(
            hakusho=HakushoSet(repo_dir="./hakushobench"),
            traces_path="./outputs/qwen38_hakusho_sft/traces.jsonl",
        )
        distill.write(rows)

    """

    def __init__(
        self,
        hakusho: HakushoSet,
        model_name: str = "Qwen/Qwen3.8-27B",
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

    def load(self) -> None:
        """
        LoRA を付けず、16bit の Qwen3.8-27B を transformers で推論用に読む.

        vLLM が無いときだけ呼ぶ.

        Returns:
            None: `model` と `processor` を属性に持たせる.

        Example:
            distill.load()

        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_name)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(
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
        複数枚の図表を思考オンで読ませ、答えが一致した短い文だけを辞書にする.

        vLLM なら並列 HTTP. transformers なら同じプロセスで generate する.

        Args:
            batch_rows (list[dict]): `question`、`answer`、`original_id`、`image` を持つ行.

        Returns:
            list[dict]: 入力と同じ順. 各要素は `original_id`、`kept`、`reason`. 残すときは `question`、`answer`、`completion` も持つ.

        Example:
            traced_rows = distill.records(rows[:4])

        """
        if self.backend == "vllm":
            with ThreadPoolExecutor(max_workers=len(batch_rows)) as pool:
                return list(pool.map(self._vllm_one, batch_rows))
        return self._hf_records(batch_rows)

    def _hf_records(self, batch_rows: list[dict]) -> list[dict]:
        """
        transformers で複数枚をまとめて生成し、採否の辞書にする.

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
            texts.append(
                processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=True,
                    reasoning_effort="low",
                )
            )
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
        raw = self._vllm_reply(chart, question)
        return self._trace(row, raw)

    def _vllm_reply(self, chart, question: str) -> str:
        """
        図表と質問を vLLM の chat completions に送り、思考と回答を連結して返す.

        Args:
            chart (PIL.Image.Image): RGB の図表.
            question (str): HakushoBench の質問.

        Returns:
            str: 思考タグと、その後の本文. サーバが思考を分けて返したときも同じ形にする.

        Example:
            raw = distill._vllm_reply(chart, question="4位の国はどこか。")

        """
        payload = {
            "model": self.model_name,
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
                "enable_thinking": True,
                "reasoning_effort": "low",
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
            url = distill._png_data_url(chart)

        """
        buffer = BytesIO()
        chart.save(buffer, format="PNG")
        encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _trace(self, row: dict, raw: str) -> dict:
        """
        生成文を採点し、jsonl に書く辞書にする.

        Args:
            row (dict): `question`、`answer`、`original_id` を持つ 1 行.
            raw (str): モデル出力. 思考タグを含み得る.

        Returns:
            dict: `original_id`、`kept`、`reason`. 残すときは `question`、`answer`、`completion` も持つ.

        Example:
            traced = distill._trace(rows[0], "答え: 3.4倍\\n")

        """
        question = str(row["question"]).strip()
        gold = str(row["answer"]).strip()
        completion, reason = self.readout.judge(raw, gold, question=question)
        traced = {
            "original_id": str(row["original_id"]),
            "kept": reason == "kept",
            "reason": reason,
        }
        if completion is None:
            return traced
        traced["question"] = question
        traced["answer"] = gold
        traced["completion"] = completion
        return traced

    def write(self, rows) -> Counter:
        """
        学習用の図表を batch_size 枚ずつ生成し、結果を jsonl の末尾に足す.

        jsonl に既にある original_id は飛ばす. 空の質問や正解も飛ばす.

        Args:
            rows (object): `original_id` を持つ学習用の図表.

        Returns:
            Counter: この実行で足した `kept`、`no_answer`、`mismatch`、`too_long`、`unsorted`、`other_as_country` の件数.

        Example:
            counts = distill.write(rows)

        """
        done = self._done_ids()
        counts = Counter()
        pending = []
        pending_index = []
        total = len(rows)
        self.traces_path.parent.mkdir(parents=True, exist_ok=True)
        with self.traces_path.open("a", encoding="utf-8") as traces:
            for index, row in enumerate(rows, start=1):
                question = str(row["question"]).strip()
                gold = str(row["answer"]).strip()
                original_id = str(row["original_id"])
                if not question or not gold or original_id in done:
                    continue
                pending.append(row)
                pending_index.append(index)
                if len(pending) < self.batch_size:
                    continue
                self._write_batch(pending, pending_index, traces, counts, total)
                pending = []
                pending_index = []
            if pending:
                self._write_batch(pending, pending_index, traces, counts, total)
        return counts

    def _write_batch(
        self,
        batch_rows: list[dict],
        indexes: list[int],
        traces,
        counts: Counter,
        total: int,
    ) -> None:
        """
        1 バッチを生成し、jsonl と件数と進捗表示に足す.

        Args:
            batch_rows (list[dict]): このバッチの図表.
            indexes (list[int]): 各図表の 1 始まりの番号.
            traces (object): 追記用に開いた jsonl.
            counts (Counter): 理由ごとの件数. このメソッドが更新する.
            total (int): 学習用の図表の総数.

        Returns:
            None: jsonl に足し、進捗を表示する.

        Example:
            distill._write_batch(pending, indexes, traces, counts, 912)

        """
        traced_rows = self.records(batch_rows)
        for index, traced in zip(indexes, traced_rows):
            traces.write(json.dumps(traced, ensure_ascii=False) + "\n")
            traces.flush()
            counts[traced["reason"]] += 1
            print(
                f"{index}/{total} {traced['reason']} "
                f"kept {counts['kept']} seen {sum(counts.values())}",
                flush=True,
            )

    def _done_ids(self) -> set[str]:
        """
        jsonl に既にある original_id を返す.

        Returns:
            set[str]: 生成を飛ばす id. ファイルが無ければ空.

        Example:
            done = distill._done_ids()

        """
        if not self.traces_path.is_file():
            return set()
        done = set()
        with self.traces_path.open(encoding="utf-8") as traces:
            for line in traces:
                if not line.strip():
                    continue
                done.add(str(json.loads(line)["original_id"]))
        return done


def main():
    parser = argparse.ArgumentParser(
        description="蒸留用の教師文を作る. 思考オンで書かせ、正解と一致した短い文だけを残す."
    )
    parser.add_argument(
        "--hakusho", required=True, help="clone した hakushobench のパス."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
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
        help="残した文と捨てた理由の jsonl.",
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
        help="Qwen 用 vLLM（serve_vllm.sh qwen）の待ち受け URL.",
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
    if args.batch_size is None:
        batch_size = 16 if backend == "vllm" else 4
    else:
        batch_size = args.batch_size
    distill.backend = backend
    distill.batch_size = batch_size
    print("backend", backend, "batch_size", batch_size)
    if backend == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        distill.load()
    counts = distill.write(train_dataset)
    print("wrote", distill.traces_path.resolve())
    print(
        "kept",
        counts["kept"],
        "no_answer",
        counts["no_answer"],
        "mismatch",
        counts["mismatch"],
        "too_long",
        counts["too_long"],
        "unsorted",
        counts["unsorted"],
        "other_as_country",
        counts["other_as_country"],
    )


if __name__ == "__main__":
    main()
