"""Gemma 4 を思考オフで 1 枚ずつ読む."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora.model import LoraModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
except ImportError:
    Gemma4ClippableLinear = None

_LORA_CREATE = LoraModel._create_and_replace


def _lora_create_clippable(
    lora_model,
    peft_config,
    adapter_name,
    target,
    target_name,
    parent,
    current_key=None,
    **kwargs,
):
    """
    Gemma4ClippableLinear は PEFT が扱えないので、内側の Linear に LoRA を付ける.

    Args:
        lora_model (object): PEFT の LoraModel.
        peft_config (object): LoRA の設定.
        adapter_name (str): アダプタ名.
        target (object): 付け先の層.
        target_name (str): 付け先の属性名.
        parent (object): 付け先の親モジュール.
        current_key (str | None): 層のパス. デフォルトは None.
        **kwargs: PEFT が渡す残りの引数.

    Returns:
        None: 内側の Linear、または元の層に LoRA を注入する.

    Example:
        LoraModel._create_and_replace = _lora_create_clippable

    """
    if Gemma4ClippableLinear is not None and isinstance(target, Gemma4ClippableLinear):
        return _LORA_CREATE(
            lora_model,
            peft_config,
            adapter_name,
            target.linear,
            "linear",
            target,
            current_key=current_key,
            **kwargs,
        )
    return _LORA_CREATE(
        lora_model,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key=current_key,
        **kwargs,
    )


class GemmaOff:
    """
    Gemma 4 を思考オフで読み、短い本文だけを返す.

    ベースだけ、または保存した LoRA を付けて generate する.

    Attributes:
        model_name (str): Hugging Face 上の Gemma 4 31B-it.
        max_new_tokens (int): 生成トークンの上限.
        model (object | None): 読み込んだモデル.
        processor (object | None): 同じ呼び出しの processor.

    Methods:
        load: ベース、または LoRA 付きを読む.
        generate: 1 枚の図表とプロンプトから本文を返す.
        close: GPU からモデルを外す.

    Example:
        gemma = GemmaOff(model_name="google/gemma-4-31B-it")
        gemma.load()
        raw = gemma.generate(chart, prompt)

    """

    soft_tokens = 1120

    def __init__(
        self,
        model_name: str = "google/gemma-4-31B-it",
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self, adapter_dir: str | Path | None = None) -> None:
        """
        16bit の Gemma 4 を読む. adapter_dir があれば LoRA を付ける.

        Args:
            adapter_dir (str | Path | None): `sft.py` が書いた LoRA. 無ければベース.

        Returns:
            None: `model` と `processor` を属性に持たせる.

        Example:
            gemma.load(adapter_dir="./outputs/gemma4_hakusho_sft")

        """
        self.close()
        processor = AutoProcessor.from_pretrained(self.model_name)
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except TypeError:
            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if adapter_dir is not None:
            adapter_path = Path(adapter_dir)
            if not (adapter_path / "adapter_config.json").is_file():
                raise SystemExit(f"LoRA がありません: {adapter_path}")
            LoraModel._create_and_replace = _lora_create_clippable
            try:
                model = PeftModel.from_pretrained(model, str(adapter_path))
            finally:
                LoraModel._create_and_replace = _LORA_CREATE
        model.eval()
        self.processor = processor
        self.model = model

    def generate(self, chart, prompt: str) -> str:
        """
        図表とユーザー文から、思考オフの本文を返す.

        Args:
            chart (PIL.Image.Image): RGB の図表.
            prompt (str): 読み取り型のユーザー文.

        Returns:
            str: 生成本文.

        Example:
            raw = gemma.generate(chart, prompt)

        """
        if self.model is None or self.processor is None:
            raise RuntimeError("先に load してください.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_kw = {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": False,
            "max_soft_tokens": self.soft_tokens,
        }
        try:
            text = self.processor.apply_chat_template(messages, **template_kw)
        except TypeError:
            template_kw.pop("max_soft_tokens")
            text = self.processor.apply_chat_template(messages, **template_kw)
        try:
            inputs = self.processor(
                chart,
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )
        except TypeError:
            inputs = self.processor(
                text=text,
                images=chart,
                return_tensors="pt",
            )
        device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in dict(inputs).items()
            }
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        generated = output_ids[0][prompt_len:]
        raw = self.processor.decode(generated, skip_special_tokens=False)
        parsed = None
        parse_response = getattr(self.processor, "parse_response", None)
        if parse_response is not None:
            try:
                parsed = parse_response(raw, prefix="")
            except TypeError:
                try:
                    parsed = parse_response(raw)
                except TypeError:
                    parsed = None
        if isinstance(parsed, dict):
            answer = parsed.get("answer") or parsed.get("content")
            if answer:
                return str(answer).strip()
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        """
        モデルを外し、GPU メモリを返す.

        Returns:
            None: 属性を空にする.

        Example:
            gemma.close()

        """
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
