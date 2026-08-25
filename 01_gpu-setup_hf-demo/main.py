import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.6-27B"
PROMPT = "こんにちは。自己紹介をしてください。"
MAX_NEW_TOKENS = 128


def memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def main():
    # モデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # プロンプトの適用
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # 入力のトークン化
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    # CUDA 使用状況を確認
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = outputs[0][prompt_tokens:]
    num_tokens = len(generated)
    reply = tokenizer.decode(generated, skip_special_tokens=True)

    print(reply)
    print()
    print(f"トークン数: {num_tokens}")
    print(f"推論時間: {elapsed:.2f}s")
    print(f"推論速度: {num_tokens / elapsed:.1f} t/s")
    if torch.cuda.is_available():
        print(f"最大 GPU メモリ使用量: {memory_gb():.2f} GiB")


if __name__ == "__main__":
    # CUDA 使用状況を確認
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("CUDA を使用して推論を実行します。")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA が使えないため推論を中止します。")
    # 推論の実行
    main()
