import gc
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

MUSE_REASONING_STRENGTH = "low"
QWEN38_REASONING_EFFORT = "low"
_MUSE_USER_RE = re.compile(
    r"to=user\s*<\|message\|>(.*?)(?:<\|eot\|>|<\|eom\|>|<\|end_of_text\|>|$)",
    re.DOTALL,
)
_MUSE_MESSAGE_RE = re.compile(
    r"<\|message\|>(.*?)(?:<\|eot\|>|<\|eom\|>|<\|end_of_text\|>|$)",
    re.DOTALL,
)

MODEL_NAMES = [
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.6-27B",
    "google/gemma-4-31B-it",
    "meta-models/Muse-Glimmer-30B",
]
OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
MAX_NEW_TOKENS = 8192

REPAIR_PROMPT_TEMPLATE = """
あなたは優秀なPython / Gradioエンジニアです。
以下に、日本株の時価総額Top10を表示するGradioアプリのベースコードがあります。
このコードに対して、以下の【修正・改善要件】をすべて反映した完成版のPythonコードを作成してください。

スクリプト名は {script_name} です。説明文やマークダウンの解説は含めず、完成した実行可能なPythonコードブロックのみを出力してください。

【修正・改善要件】
1. **金額・単位の明記とフォーマット**
   - 株価は通貨単位が「円」（またはJPY）であることを明記し、3桁ごとにカンマ区切りを入れること
     （例: `3,250 円` または 列名を `株価（円）` にする）。
   - 時価総額は「兆円」単位（小数第2位程度）に換算して明記し、3桁ごとにカンマ区切りを入れること
     （例: `35.42 兆円` または 列名を `時価総額（兆円）` にする）。
   - 【最重要】表示用文字列にフォーマットする前に、生の数値データで時価総額の降順ソートを行うこと
     （文字列ソートで順位が壊れないようにすること）。

2. **起動・読み込みタイミングの改善**
   - アプリ起動前（インポート時・定義時）に重い通信を行わず、UI起動後（`app.load` / `demo.load`）に
     データを取得すること。

3. **UI / UX の強化とデザイン**
   - ユーザーにとって見やすい色味や統一感のあるデザインにしてください
     （Gradioのテーマ設定、カラースキーム、見やすいCSSやマークダウンの工夫など）。
   - 表の上に「YYYY-MM-DD HH:MM:SS JST 時点」の取得日時と、取得件数やステータスを表示すること。
   - 「更新」ボタンを目立たせ、使いやすい配置にすること（variant='primary' など）。
   - データフレームは閲覧専用として見やすく表示すること（interactive=False など）。

4. **制約事項（厳守）**
   - 候補ユニバースの銘柄一覧（31銘柄）および日本語名の対応はベースコードのものをそのまま正しく維持すること（増減・推測禁止）。
   - 取得失敗した銘柄は除外すること。

----- 現在のベースコード -----
{base_code}
----- ベースコードここまで -----
""".strip()


def memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def extract_python_script(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip() + "\n"
    return text.strip() + "\n"


def output_path_for(model_name: str, *, thinking: bool | None = None) -> Path:
    short_name = model_name.rsplit("/", 1)[-1]
    if thinking is True:
        return OUTPUT_DIR / f"topix_{short_name}_thinking-on_repaired.py"
    if thinking is False:
        return OUTPUT_DIR / f"topix_{short_name}_thinking-off_repaired.py"
    return OUTPUT_DIR / f"topix_{short_name}_repaired.py"


def find_existing_code(
    model_name: str, *, thinking: bool | None = None
) -> tuple[str | None, Path | None]:
    short_name = model_name.rsplit("/", 1)[-1]
    candidates = []
    if thinking is True:
        candidates.extend(
            [
                OUTPUT_DIR / f"topix_{short_name}_thinking-on.py",
                PROJECT_ROOT / f"topix_{short_name}_thinking-on.py",
            ]
        )
    elif thinking is False:
        candidates.extend(
            [
                OUTPUT_DIR / f"topix_{short_name}_thinking-off.py",
                PROJECT_ROOT / f"topix_{short_name}_thinking-off.py",
            ]
        )
    candidates.extend(
        [
            OUTPUT_DIR / f"topix_{short_name}.py",
            PROJECT_ROOT / f"topix_{short_name}.py",
            PROJECT_ROOT / "result" / f"topix_{short_name}.py",
        ]
    )

    for p in candidates:
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    return content, p
            except Exception:
                pass

    return None, None


def is_muse(model_name: str) -> bool:
    return "muse" in model_name.lower()


def is_qwen38(model_name: str) -> bool:
    return "qwen3.8" in model_name.lower()


def strip_qwen_thinking(text: str) -> str:
    """Qwen の思考ブロックを除く。開始 <think> は special token として消えることがある。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        text = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
    return text.strip()


def parse_muse_reply(text: str) -> str:
    user_match = _MUSE_USER_RE.search(text)
    if user_match:
        return user_match.group(1).strip()
    messages = _MUSE_MESSAGE_RE.findall(text)
    if messages:
        return messages[-1].strip()
    return text.strip()


def apply_chat_template(
    tokenizer, prompt: str, model_name: str, *, enable_thinking: bool | None = None
) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if is_qwen38(model_name):
        thinking = True if enable_thinking is None else enable_thinking
        kwargs["enable_thinking"] = thinking
        if thinking:
            kwargs["reasoning_effort"] = QWEN38_REASONING_EFFORT
    elif "qwen" in model_name.lower():
        kwargs["enable_thinking"] = False
    elif is_muse(model_name):
        kwargs["reasoning_strength"] = MUSE_REASONING_STRENGTH
    return tokenizer.apply_chat_template(messages, **kwargs)


def release_gpu(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_reply(
    model,
    tokenizer,
    model_name: str,
    prompt: str,
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, int, float]:
    text = apply_chat_template(
        tokenizer, prompt, model_name, enable_thinking=enable_thinking
    )
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
    }
    if is_muse(model_name):
        generate_kwargs.update(
            do_sample=False,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = outputs[0][prompt_tokens:]
    num_tokens = len(generated)
    if is_muse(model_name):
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        reply = parse_muse_reply(raw)
    else:
        reply = tokenizer.decode(generated, skip_special_tokens=True)
        if is_qwen38(model_name) and enable_thinking:
            reply = strip_qwen_thinking(reply)

    del inputs, outputs
    return reply, num_tokens, elapsed


def save_and_report(
    reply: str,
    output_path: Path,
    *,
    num_tokens: int,
    elapsed: float,
    label: str = "",
) -> None:
    script = extract_python_script(reply)
    output_path.write_text(script, encoding="utf-8")

    if label:
        print(f"[{label}]")
    print(reply)
    print()
    print(f"生成修正スクリプト: {output_path.resolve()}")
    print(f"トークン数: {num_tokens}")
    print(f"推論時間: {elapsed:.2f}s")
    print(f"推論速度: {num_tokens / elapsed:.1f} t/s")
    if torch.cuda.is_available():
        print(f"最大 GPU メモリ使用量: {memory_gb():.2f} GiB")
    print()


def run_one(model_name: str) -> None:
    print("=" * 60)
    print(f"モデル: {model_name} (リペア・改善実験)")
    print("=" * 60)

    # 実行対象タスク（thinking設定とベースコード）を事前に確認
    tasks: list[tuple[bool | None, str, str]] = []
    if is_qwen38(model_name):
        for thinking in (False, True):
            label = "thinking-on" if thinking else "thinking-off"
            base_code, path = find_existing_code(model_name, thinking=thinking)
            if base_code is None:
                print(
                    f"[失敗/スキップ] {model_name} ({label}) のベースコードが見つかりません。"
                )
            else:
                print(f"ベースコード読み込み成功 ({label}): {path}")
                tasks.append((thinking, base_code, label))
    else:
        base_code, path = find_existing_code(model_name)
        if base_code is None:
            print(f"[失敗/スキップ] {model_name} のベースコードが見つかりません。")
        else:
            print(f"ベースコード読み込み成功: {path}")
            tasks.append((None, base_code, ""))

    if not tasks:
        print(
            f"-> {model_name} のベースコードが見つからないため、"
            "このモデルは失敗として次のモデルに移ります。\n"
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model_cls = (
        AutoModelForMultimodalLM if is_muse(model_name) else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for thinking, base_code, label in tasks:
        output_path = output_path_for(model_name, thinking=thinking)
        tag = f" ({label})" if label else ""
        print(f"出力{tag}: {output_path}")
        prompt = REPAIR_PROMPT_TEMPLATE.format(
            script_name=output_path.name,
            base_code=base_code,
        )
        reply, num_tokens, elapsed = generate_reply(
            model,
            tokenizer,
            model_name,
            prompt,
            enable_thinking=thinking,
        )
        save_and_report(
            reply,
            output_path,
            num_tokens=num_tokens,
            elapsed=elapsed,
            label=label,
        )

    release_gpu(model, tokenizer)


def main():
    for model_name in MODEL_NAMES:
        run_one(model_name)


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("CUDA を使用してリペア推論を実行します。")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA が使えないため推論を中止します。")
    main()
