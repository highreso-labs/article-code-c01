import gc
import re
import ssl
import time
import urllib.error
import urllib.request
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
    # "Qwen/Qwen3.8-27B",
    # "Qwen/Qwen3.6-27B",
    # "google/gemma-4-31B-it",
    "meta-models/Muse-Glimmer-30B",
]
OUTPUT_DIR = Path(__file__).resolve().parent
ARTICLE_URL = "https://ornith.ai/ornith_1_5.html"
ARTICLE_HTML_PATH = OUTPUT_DIR / "ornith_1_5.html"
MAX_NEW_TOKENS = 4096

PROMPT_TEMPLATE = """
次のHTML記事を読んだ。LLMに詳しくない現場の人（開発・企画・運用）が、これを読んで日々の業務になんらかの活用したいと考えている。日本語で解説してほしい。
わかりやすさと親切さが最優先。専門用語は出したらすぐ日常のたとえを付ける。結論を先に書く。
数字は正確に使うこと。必要なら図表を用いても良い。
出力ファイル名は {output_name}。完成したMarkdownだけを出力すること。

含めてほしいこと:
- 一言で何か、何がどのようにすごいのか？
- 誰が作ったか（記事から分かることだけ。分からなければ分からないと書く）
- 何が新しいか（たとえ付き。報酬の式や学習の仕組みも、現場向けに噛み砕いてよい）
- これらの情報から現場ではどのような活用方法が考えられるか？
- 課題となりそうな点や、注意点はなにか？

記事にない人名、所属、点数は足さない。
最後に参考URLを書くこと。

出典: https://ornith.ai/ornith_1_5.html

----- 記事HTMLここから -----
{article}
----- 記事HTMLここまで -----
""".strip()


def download_article_html() -> str:
    request = urllib.request.Request(
        ARTICLE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except (ssl.SSLError, urllib.error.URLError):
        # ornith.ai の証明書が切れている場合がある
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=30, context=context) as resp:
            return resp.read().decode("utf-8")


def load_article_html() -> str:
    print(f"記事を取得: {ARTICLE_URL}")
    try:
        html = download_article_html()
        ARTICLE_HTML_PATH.write_text(html, encoding="utf-8")
        print(f"保存: {ARTICLE_HTML_PATH}")
        return html
    except Exception as exc:
        if ARTICLE_HTML_PATH.exists():
            print(f"取得失敗 ({exc})。ローカル {ARTICLE_HTML_PATH} を使う")
            return ARTICLE_HTML_PATH.read_text(encoding="utf-8")
        raise SystemExit(f"記事HTMLを取得できません: {ARTICLE_URL}\n{exc}") from exc


def html_content_for_prompt(html: str) -> str:
    """CSS/JSは記事本体ではないので除く。本文・数式・表・脚注は残す。"""
    html = re.sub(
        r"<script\b[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL
    )
    html = re.sub(
        r"<style\b[^>]*>.*?</style>", "", html, flags=re.IGNORECASE | re.DOTALL
    )
    return html.strip()


def build_prompt(output_name: str, article: str) -> str:
    # HTML内の { }（MathJax）を str.format が壊さないよう、置換だけ使う
    return PROMPT_TEMPLATE.replace("{output_name}", output_name).replace(
        "{article}", article
    )


def memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def extract_markdown(text: str) -> str:
    """本文中の ``` コードブロックは触らず、文書全体を包む ```markdown だけ外す。"""
    text = text.strip()
    outer = re.match(r"```(?:markdown|md)\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if outer:
        return outer.group(1).strip() + "\n"
    return text + "\n"


def output_path_for(model_name: str) -> Path:
    short_name = model_name.rsplit("/", 1)[-1]
    return OUTPUT_DIR / f"research_ornith_{short_name}.md"


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


def apply_chat_template(tokenizer, prompt: str, model_name: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if is_qwen38(model_name):
        kwargs["enable_thinking"] = True
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


def run_one(model_name: str, article: str) -> None:
    output_path = output_path_for(model_name)
    prompt = build_prompt(output_path.name, article)

    print("=" * 60)
    print(f"モデル: {model_name}")
    print(f"出力: {output_path}")
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

    text = apply_chat_template(tokenizer, prompt, model_name)
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
        if is_qwen38(model_name):
            reply = strip_qwen_thinking(reply)
    markdown = extract_markdown(reply)
    output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    print()
    print(f"生成Markdown: {output_path.resolve()}")
    print(f"トークン数: {num_tokens}")
    print(f"推論時間: {elapsed:.2f}s")
    print(f"推論速度: {num_tokens / elapsed:.1f} t/s")
    if torch.cuda.is_available():
        print(f"最大 GPU メモリ使用量: {memory_gb():.2f} GiB")
    print()

    release_gpu(model, tokenizer, inputs, outputs)


def main():
    article = html_content_for_prompt(load_article_html())
    print(f"記事HTML: {ARTICLE_HTML_PATH} ({len(article):,} 文字)")
    for model_name in MODEL_NAMES:
        run_one(model_name, article)


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("CUDA を使用して推論を実行します。")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA が使えないため推論を中止します。")
    main()
