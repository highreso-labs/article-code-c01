"""自己生成（Gemma 4 E4B）と教師補完（Qwen3.8）の思考付きデータで、Gemma 4 E4B-it の 16bit LoRA を SFT する.

思考モード（`<|turn>system\\n<|think|><turn|>` と `<|channel>thought...<channel|>`）を有効にし、
日本語での段階的読み取りプロセスと「答え:」の行を合わせて学習する.
言語層の Attention と MLP の両方に LoRA（rank 16）を付ける.

使い方:
    python sft.py --hakusho /path/to/hakushobench
"""

from __future__ import annotations

import argparse
import functools
import json
import time
import warnings
from collections import Counter
from pathlib import Path

from hakusho import HakushoSet

warnings.filterwarnings("ignore", message="Accessing `prepare_peft_model` from")
from unsloth import FastModel, FastVisionModel
from unsloth.chat_templates import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator

from datasets import Dataset, Image as ArrowImage
from trl import SFTConfig, SFTTrainer

from readout import ChartReadout
from report import TrainReport


def _wrap_thinking_chat_template(apply_fn, soft_tokens: int):
    """
    `apply_chat_template` に既定で `enable_thinking=True` と `max_soft_tokens` を渡すラッパーを返す.

    Args:
        apply_fn (callable): 元の `apply_chat_template`.
        soft_tokens (int): 画像ソフトトークンの上限.

    Returns:
        callable: 思考オンを既定にした `apply_chat_template`.

    Example:
        tokenizer.apply_chat_template = _wrap_thinking_chat_template(tokenizer.apply_chat_template, 1120)

    """

    @functools.wraps(apply_fn)
    def wrapped(conversation, *args, **kwargs):
        kwargs.setdefault("enable_thinking", True)
        kwargs.setdefault("max_soft_tokens", soft_tokens)
        try:
            return apply_fn(conversation, *args, **kwargs)
        except TypeError:
            kwargs.pop("max_soft_tokens", None)
            try:
                return apply_fn(conversation, *args, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return apply_fn(conversation, *args, **kwargs)

    return wrapped


class Gemma4SFT:
    """
    思考＋根拠＋「答え:」のトレースで、言語層の Attention と MLP を LoRA SFT する.

    生徒（Gemma 4 E4B）の自力正解トレースと、教師（Qwen3.8）の補完トレースを
    Gemma 4 の思考モード形式に統一して学習する.

    Attributes:
        SOFT_TOKENS (int): 画像ソフトトークンの上限. 1120.
        LORA_RANK (int): LoRA の rank と alpha. 16.
        hakusho (HakushoSet): 図表を PIL にする読み手.
        model_name (str): Hugging Face 上の Gemma 4 E4B-it.
        max_seq_length (int): プロンプトと思考・回答を合わせた最大トークン長.
        seed (int): LoRA と学習の乱数.
        output_dir (Path): LoRA と tokenizer の保存先.

    Methods:
        load: 16bit モデルに Attention + MLP の LoRA を付ける.
        join: jsonl の残した思考＋本文と、学習用の図表を結合する.
        to_messages: 1 件を Unsloth の画像付き思考オン messages にする.
        fit: SFT し、LoRA と経過時間を返す.

    Example:
        sft = Gemma4SFT(
            hakusho=HakushoSet(repo_dir="./hakushobench"),
            output_dir="./outputs/gemma4_hakusho_sft",
        )
        model, tokenizer = sft.load()
        sft.fit(model, tokenizer, train_dataset)

    """

    SOFT_TOKENS = 1120
    LORA_RANK = 8

    def __init__(
        self,
        hakusho: HakushoSet,
        model_name: str = "google/gemma-4-E4B-it",
        max_seq_length: int = 8192,
        seed: int = HakushoSet.SEED,
        output_dir: str | Path = "outputs/gemma4_hakusho_sft",
    ):
        self.hakusho = hakusho
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.readout = ChartReadout()
        self._use_system_think_fallback = False
        self._use_thinking_field = False

    def load(self):
        """
        16bit の Gemma 4 E4B-it を読み、言語層の Attention と MLP に LoRA（rank 16）を付ける.

        チャットテンプレートは `enable_thinking=True` を有効化する.
        vision 層は `Gemma4ClippableLinear` と vLLM `--enable-lora` の互換性のため凍結する.

        Returns:
            tuple[object, object]: 学習用モデルと、思考オン設定済みの tokenizer/processor.

        Example:
            model, tokenizer = sft.load()

        """
        model, tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            fast_inference=False,
            use_exact_model_name=True,
        )
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        tokenizer.apply_chat_template = _wrap_thinking_chat_template(
            tokenizer.apply_chat_template,
            self.SOFT_TOKENS,
        )
        inner_tok = getattr(tokenizer, "tokenizer", None)
        if inner_tok is not None and hasattr(inner_tok, "apply_chat_template"):
            inner_tok.apply_chat_template = _wrap_thinking_chat_template(
                inner_tok.apply_chat_template,
                self.SOFT_TOKENS,
            )
        probe_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Q"}]},
            {
                "role": "assistant",
                "thinking": "THOUGHT_PROBE",
                "content": [{"type": "text", "text": "ANS_PROBE"}],
            },
        ]
        rendered_probe = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        self._use_system_think_fallback = "<|think|>" not in rendered_probe
        self._use_thinking_field = (
            "<|channel>thought\nTHOUGHT_PROBE\n<channel|>ANS_PROBE" in rendered_probe
        )
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=False,
            r=self.LORA_RANK,
            lora_alpha=self.LORA_RANK,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None,
        )
        return model, tokenizer

    def join(
        self, traces_path: Path, train_dataset: Dataset
    ) -> tuple[Dataset, Counter]:
        """
        jsonl で残した思考＋本文に、同じ original_id の図表を結び、SFT 用の Dataset にする.

        プロンプトは今の読み取り型で作り直す. 学習用に無い id は使わない.

        Args:
            traces_path (Path): `distill_data.py` が書いた jsonl.
            train_dataset (Dataset): `split_dataset` が返した学習用の図表.

        Returns:
            tuple[Dataset, Counter]: (`image`、`prompt`、`thinking`、`completion` を持つ Dataset, 生成元モデル別の件数).

        Example:
            train_dataset, source_counts = sft.join(traces_path, train_dataset)

        """
        kept = []
        with traces_path.open(encoding="utf-8") as traces:
            for line in traces:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kept"):
                    kept.append(record)
        by_id = {str(row["original_id"]): row for row in train_dataset}
        samples = []
        source_counts = Counter()
        for record in kept:
            row = by_id.get(str(record["original_id"]))
            if row is None:
                continue
            raw_completion = str(record["completion"])
            thinking = str(record.get("thinking") or "").strip()
            if not thinking:
                thinking = self.readout.extract_thought(raw_completion)
            visible_completion = (
                self.readout.visible(raw_completion) or raw_completion.strip()
            )
            source_model = str(record.get("model") or "unknown")
            source_counts[source_model] += 1
            samples.append(
                {
                    "image": self.hakusho.load_chart(row),
                    "prompt": self.readout.prompt(str(record["question"])),
                    "thinking": thinking,
                    "completion": visible_completion,
                }
            )
        if not samples:
            raise SystemExit(
                f"SFT に使える文がありません: {traces_path}\n"
                "distill_data.py の kept が 1 件以上あることを確認してください."
            )
        return (
            Dataset.from_list(samples).cast_column("image", ArrowImage()),
            source_counts,
        )

    def to_messages(self, example: dict) -> dict:
        """
        1 件の図表と思考・本文を、Unsloth の vision collator が読む messages にする.

        Args:
            example (dict): `image`、`prompt`、`thinking`、`completion` を持つ 1 件.

        Returns:
            dict: `messages` と `images` を持つ 1 件.

        Example:
            messages = sft.to_messages(dataset[0])

        """
        messages = []
        if self._use_system_think_fallback:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "<|think|>"}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]},
                ],
            }
        )
        thinking = str(example.get("thinking") or "").strip()
        completion = str(example["completion"]).strip() + "\n"
        if self._use_thinking_field and thinking:
            messages.append(
                {
                    "role": "assistant",
                    "thinking": thinking,
                    "content": [{"type": "text", "text": completion}],
                }
            )
        else:
            assistant_text = self.readout.format_assistant(thinking, completion)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )
        return {
            "messages": messages,
            "images": [example["image"]],
        }

    def fit(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        batch_size: int = 2,
        grad_accum: int = 4,
        max_steps: int = 160,
        learning_rate: float = 2e-5,
    ):
        """
        アシスタントの思考チャネルと本文を SFT し、LoRA と tokenizer を output_dir に書く.

        損失は `<|turn>model\\n` 以降（`<|channel>thought...<channel|>` と根拠・答え）に付ける.

        Args:
            model (object): `load` が返した LoRA 付きモデル.
            tokenizer (object): 同じ呼び出しの tokenizer.
            train_dataset (Dataset): `join` が返した学習データ.
            batch_size (int): GPU あたりのバッチ. デフォルトは 2.
            grad_accum (int): 勾配蓄積. デフォルトは 4.
            max_steps (int): ステップ上限. デフォルトは 160.
            learning_rate (float): 学習率. デフォルトは 2e-5.

        Returns:
            dict: `train_result`、`train_seconds`、`save_seconds`、`log_history`. LoRA も output_dir に保存する.

        Example:
            trained = sft.fit(model, tokenizer, train_dataset)

        """
        FastVisionModel.for_training(model)
        collator = UnslothVisionDataCollator(
            model,
            tokenizer,
            max_seq_length=self.max_seq_length,
            formatting_func=self.to_messages,
            resize="max",
            train_on_responses_only=True,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            data_collator=collator,
            train_dataset=train_dataset,
            args=SFTConfig(
                output_dir=str(self.output_dir / "trainer"),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=learning_rate,
                max_steps=max_steps,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                seed=self.seed,
                report_to="none",
                save_strategy="no",
                packing=False,
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=self.max_seq_length,
            ),
        )
        train_started = time.perf_counter()
        train_result = trainer.train()
        train_seconds = time.perf_counter() - train_started
        print(train_result)
        save_started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        save_seconds = time.perf_counter() - save_started
        print("saved", self.output_dir.resolve())
        return {
            "train_result": train_result,
            "train_seconds": train_seconds,
            "save_seconds": save_seconds,
            "log_history": list(getattr(trainer.state, "log_history", [])),
        }


def main():
    parser = argparse.ArgumentParser(
        description="思考付き読み取り型で Gemma 4 E4B-it を 16bit LoRA SFT する."
    )
    parser.add_argument(
        "--hakusho", required=True, help="clone した hakushobench のパス."
    )
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "gemma4_hakusho_sft"),
        help="LoRA の保存先.",
    )
    parser.add_argument(
        "--traces",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "qwen38_hakusho_sft"
            / "traces.jsonl"
        ),
        help="distill_data.py の jsonl.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=HakushoSet.EVAL_RATIO,
        help="テストに回す割合. distill_data.py と同じ値にする.",
    )
    parser.add_argument("--seed", type=int, default=HakushoSet.SEED)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    wall_started = time.perf_counter()
    traces_path = Path(args.traces)
    if not traces_path.is_file():
        raise SystemExit(
            f"教師データがありません: {traces_path}\n"
            "先に distill_data.py を実行してください."
        )

    prep_started = time.perf_counter()
    hakusho = HakushoSet(repo_dir=args.hakusho)
    rows = hakusho.filter_charts(hakusho.load_rows())
    train_dataset, test_dataset = hakusho.split_dataset(
        rows,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    sft = Gemma4SFT(
        hakusho=hakusho,
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        output_dir=output_dir,
    )
    train_dataset, source_counts = sft.join(traces_path, train_dataset)
    prep_seconds = time.perf_counter() - prep_started
    print(
        "sft",
        len(train_dataset),
        "test",
        len(test_dataset),
        "sources",
        dict(source_counts),
    )

    load_started = time.perf_counter()
    model, tokenizer = sft.load()
    load_seconds = time.perf_counter() - load_started
    trained = sft.fit(
        model,
        tokenizer,
        train_dataset,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    total_seconds = time.perf_counter() - wall_started
    train_result = trained["train_result"]
    metrics = getattr(train_result, "metrics", None) or {}
    report = TrainReport(title="HakushoBench Thinking SFT")
    report.add("モデル", args.model)
    report.add("教師データ", str(traces_path))
    report.add("SFT 件数", len(train_dataset))
    report.add("テスト件数", f"{len(test_dataset)} 件")
    report.add("LoRA rank", Gemma4SFT.LORA_RANK)
    report.add("バッチサイズ", args.batch_size)
    report.add("勾配蓄積", args.grad_accum)
    report.add("ステップ上限", args.max_steps)
    report.add("実際のステップ", getattr(train_result, "global_step", ""))
    report.add("学習損失", metrics.get("train_loss", ""))
    report.add("データ準備", report.clock(prep_seconds))
    report.add("モデル読み込み", report.clock(load_seconds))
    report.add("学習", report.clock(trained["train_seconds"]))
    report.add("LoRA 保存", report.clock(trained["save_seconds"]))
    report.add("全体", report.clock(total_seconds))
    if source_counts:
        report.add_counts("データ生成元モデル", source_counts.most_common())
    report.show()
    report_path = report.save(output_dir / "sft_report.html")
    print("report", report_path.resolve())


if __name__ == "__main__":
    main()
