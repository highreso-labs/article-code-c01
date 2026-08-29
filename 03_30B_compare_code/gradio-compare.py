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
MAX_NEW_TOKENS = 4096

PROMPT_TEMPLATE = """
Gradioで「日本株の時価総額Top10」を表示するアプリを作ってほしい。
スクリプト名は {script_name}。完成した実行可能なPythonコードだけを出力すること。

【やってはいけないこと】
- yfinanceの ^TOPX / constituents で構成銘柄を取らない（このAPIは構成銘柄を返さない）
- 「Top10らしい銘柄」を最初から10個だけハードコードしない
- 銘柄コードと社名を推測で対応づけない（例: 6752をキヤノンと書かない）

【手順】
1. 候補は30銘柄以上のユニバースから取る（下記の確定リスト）
2. 各銘柄の時価総額を yfinance の marketCap で取得する
3. 時価総額の降順でソートし、上位10件だけ表示する
4. 銘柄名は日本語で表示する。yfinanceのlongNameは英語なので使わない
5. 銘柄名は下記リストの日本語名をコードとセットで使う（推測・改名禁止）
6. 取得失敗した銘柄は除外する
7. データ取得時点（日本時間、秒まで）を画面に表示する

【候補ユニバース（この対応をそのまま使え。増減・改名禁止）】
7203.T トヨタ自動車
8306.T 三菱UFJフィナンシャル・グループ
6758.T ソニーグループ
9984.T ソフトバンクグループ
6501.T 日立製作所
8035.T 東京エレクトロン
8316.T 三井住友フィナンシャルグループ
9983.T ファーストリテイリング
6857.T アドバンテスト
8411.T みずほフィナンシャルグループ
4063.T 信越化学工業
6098.T リクルートホールディングス
8058.T 三菱商事
8001.T 伊藤忠商事
8031.T 三井物産
6861.T キーエンス
9432.T 日本電信電話
9433.T KDDI
7011.T 三菱重工業
7974.T 任天堂
4568.T 第一三共
4519.T 中外製薬
6367.T ダイキン工業
8766.T 東京海上ホールディングス
7741.T HOYA
6902.T デンソー
4543.T テルモ
6954.T ファナック
6594.T ニデック
7267.T ホンダ
285A.T キオクシアホールディングス

【UI】
- GradioのDataframeで、順位・銘柄コード・銘柄名（日本語）・株価・時価総額を表示
- 表の上に「YYYY-MM-DD HH:MM:SS JST 時点」のように取得日時を表示する
- 起動時に1回取得し、更新ボタンでも再取得できること。更新時は日時も更新する
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
        return OUTPUT_DIR / f"topix_{short_name}_thinking-on.py"
    if thinking is False:
        return OUTPUT_DIR / f"topix_{short_name}_thinking-off.py"
    return OUTPUT_DIR / f"topix_{short_name}.py"


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
    print(f"生成スクリプト: {output_path.resolve()}")
    print(f"トークン数: {num_tokens}")
    print(f"推論時間: {elapsed:.2f}s")
    print(f"推論速度: {num_tokens / elapsed:.1f} t/s")
    if torch.cuda.is_available():
        print(f"最大 GPU メモリ使用量: {memory_gb():.2f} GiB")
    print()


def run_one(model_name: str) -> None:
    print("=" * 60)
    print(f"モデル: {model_name}")
    print("=" * 60)

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

    if is_qwen38(model_name):
        for thinking in (False, True):
            output_path = output_path_for(model_name, thinking=thinking)
            label = "thinking-on" if thinking else "thinking-off"
            print(f"出力 ({label}): {output_path}")
            prompt = PROMPT_TEMPLATE.format(script_name=output_path.name)
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
    else:
        output_path = output_path_for(model_name)
        print(f"出力: {output_path}")
        prompt = PROMPT_TEMPLATE.format(script_name=output_path.name)
        reply, num_tokens, elapsed = generate_reply(
            model, tokenizer, model_name, prompt
        )
        save_and_report(reply, output_path, num_tokens=num_tokens, elapsed=elapsed)

    release_gpu(model, tokenizer)


def main():
    for model_name in MODEL_NAMES:
        run_one(model_name)


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("CUDA を使用して推論を実行します。")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA が使えないため推論を中止します。")
    main()
