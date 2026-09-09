import gc
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = AutoModelForMultimodalLM

MUSE_REASONING_STRENGTH = "low"
QWEN38_REASONING_EFFORT = "low"
GEMMA_MAX_SOFT_TOKENS = 1120
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
PDF_URL = "https://www.meti.go.jp/report/tsuhaku2026/pdf/1-1-1.pdf"
PDF_FILENAME = "1-1-1.pdf"
WGET_CMD = (
    f'wget --user-agent="Mozilla/5.0" -O {PDF_FILENAME} "{PDF_URL}"'
)
PAGE_DIR = OUTPUT_DIR / "_pdf_pages"
MAX_NEW_TOKENS = 1536
MAX_PAGES = 12
RENDER_DPI = 200

SHARED_RULES = """
添付画像はPDFの全ページを1ページ目から順にラスター化したものである。

【回答形式】
- 与えた表だけを出力する。前置き・解説・根拠は書かない
- ID列と問い列はそのまま写し、「答え」列だけを埋める
- 答えは求められた値ひとつだけ（例: 19.4% / ベネズエラ / 1,787）
- 単位を聞かれた項目以外は、単位や括弧書きの補足を足さない
- 本文の記述ではなく、指定された図表そのものを読む
- 一般知識で補わない。PDFに書かれた値をそのまま写す
- 読めないときだけ「記載なし」と書く
""".strip()

QUESTIONS = [
    {
        "id": "q1",
        "title": "問1　表のセル",
        "prompt": """
第1-1-1-5表と第1-1-1-8表から、指定した行と列が交わるセルを読め。

| ID | 問い | 答え |
|---|---|---|
| t1 | 第1-1-1-5表 イラクの世界の名目GDPにおける順位 | |
| t2 | 第1-1-1-5表 バーレーンの名目GDP（10億ドル） | |
| t3 | 第1-1-1-5表 イエメンの世界の名目GDPにおける順位 | |
| t4 | 第1-1-1-5表 レバノンが世界の名目GDPに占める割合 | |
| t5 | 第1-1-1-5表 に並ぶ中東諸国の数（地域計の行は除く） | |
| t6 | 第1-1-1-5表 備考が挙げる構成国の分類元 | |
| t7 | 第1-1-1-8表 サウジアラビアのOECD加盟アジア･大洋州向け | |
| t8 | 第1-1-1-8表 イラクのインド向け | |
| t9 | 第1-1-1-8表 イランの中国向け | |
| t10 | 第1-1-1-8表 クウェートの中東向け | |
| t11 | 第1-1-1-8表 アラブ首長国連邦の合計 | |
| t12 | 第1-1-1-8表 の数値の単位 | |
| t13 | 第1-1-1-8表 備考の網掛け基準（シェアが何%を超える部分か） | |
""".strip(),
    },
    {
        "id": "q2",
        "title": "問2　グラフの軸と注記",
        "prompt": """
折れ線グラフの軸目盛りと図中の注記を読め。目盛りは軸に印字されている値をそのまま答える。

| ID | 問い | 答え |
|---|---|---|
| l1 | 第1-1-1-2図 に並ぶパネルの数 | |
| l2 | 第1-1-1-2図 横軸の最終年 | |
| l3 | 第1-1-1-2図「日本」パネルの縦軸の最大目盛り | |
| l4 | 第1-1-1-2図「米国」パネルの縦軸の最大目盛り | |
| l5 | 第1-1-1-2図「EU」パネルの縦軸の最大目盛り | |
| l6 | 第1-1-1-2図「新興国」パネルの縦軸の最小目盛り | |
| l7 | 第1-1-1-2図「インド」パネルの縦軸の最小目盛り | |
| l8 | 第1-1-1-3図 縦軸の最大目盛り | |
| l9 | 第1-1-1-3図 横軸の開始年 | |
| l10 | 第1-1-1-3図 に注記された関税・協定の数 | |
| l11 | 第1-1-1-3図 Morrill tariff の年 | |
| l12 | 第1-1-1-4図 縦軸の最大目盛り | |
| l13 | 第1-1-1-11図 縦軸の最大目盛り | |
| l14 | 第1-1-1-11図 価格の単位 | |
""".strip(),
    },
    {
        "id": "q3",
        "title": "問3　円グラフの内訳",
        "prompt": """
円グラフを読め。順位はシェアの大きい順とする。総量は円の中心に書かれている値を単位まで答える。

| ID | 問い | 答え |
|---|---|---|
| p1 | 第1-1-1-6図 埋蔵量 1位の国 | |
| p2 | 第1-1-1-6図 埋蔵量 1位のシェア | |
| p3 | 第1-1-1-6図 埋蔵量の総量 | |
| p4 | 第1-1-1-6図 生産量 1位の国 | |
| p5 | 第1-1-1-6図 生産量の総量 | |
| p6 | 第1-1-1-6図 輸出量 2位の国 | |
| p7 | 第1-1-1-7図 埋蔵量 1位の国 | |
| p8 | 第1-1-1-7図 埋蔵量の総量 | |
| p9 | 第1-1-1-7図 生産量 1位の国 | |
| p10 | 第1-1-1-7図 輸出量の総量 | |
| p11 | 第1-1-1-9図 日本の1位の相手国 | |
| p12 | 第1-1-1-9図 ベトナムの1位のシェア | |
| p13 | 第1-1-1-9図 インドネシアの1位の相手国 | |
| p14 | 第1-1-1-10図 日本の1位の相手国 | |
| p15 | 第1-1-1-10図 タイの4位の相手国 | |
""".strip(),
    },
]


def memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def extract_markdown(text: str) -> str:
    text = text.strip()
    outer = re.match(r"```(?:markdown|md)\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if outer:
        return outer.group(1).strip() + "\n"
    return text + "\n"


def build_question_prompt(question: dict) -> str:
    return SHARED_RULES + "\n\n" + question["prompt"] + "\n"


def output_path_for(model_name: str, *, thinking: bool | None = None) -> Path:
    short_name = model_name.rsplit("/", 1)[-1]
    if thinking is True:
        return OUTPUT_DIR / f"pdf_{short_name}_thinking-on.md"
    if thinking is False:
        return OUTPUT_DIR / f"pdf_{short_name}_thinking-off.md"
    return OUTPUT_DIR / f"pdf_{short_name}.md"


def is_muse(model_name: str) -> bool:
    return "muse" in model_name.lower()


def is_qwen38(model_name: str) -> bool:
    return "qwen3.8" in model_name.lower()


def is_gemma(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def strip_qwen_thinking(text: str) -> str:
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


def resolve_pdf_path() -> Path:
    candidates = [
        Path.cwd() / PDF_FILENAME,
        OUTPUT_DIR / PDF_FILENAME,
        Path.cwd() / "tsuhaku2026_1-1-1.pdf",
        OUTPUT_DIR / "tsuhaku2026_1-1-1.pdf",
    ]
    found: Path | None = None
    for path in candidates:
        if path.exists():
            found = path
            break
    if found is None:
        raise SystemExit(
            f"ローカルPDFが見つかりません。{PDF_FILENAME} をカレントディレクトリに置いてください。\n"
            f"例:\n  {WGET_CMD}"
        )
    size = found.stat().st_size
    header = found.read_bytes()[:8]
    if size == 0 or not header.startswith(b"%PDF"):
        raise SystemExit(
            f"{found} は空か、PDFではありません（{size} bytes）。"
            f"403で空ファイルになっていることが多いです。削除してから User-Agent 付きで取り直してください。\n"
            f"  rm {found.name}\n"
            f"  {WGET_CMD}\n"
            f"それでもダメなら手元で保存して scp してください。"
        )
    return found


def render_pdf_pages(pdf_path: Path) -> list[Image.Image]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("PDFのページ画像化には pymupdf が必要です: pip install pymupdf") from exc

    doc = fitz.open(pdf_path)
    total = doc.page_count
    zoom = RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    limit = min(total, MAX_PAGES)
    for i in range(limit):
        pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    if not images:
        raise SystemExit(f"PDFからページを読めません: {pdf_path}")
    print(f"ページ数: {total}（推論に使う: {len(images)} / MAX_PAGES={MAX_PAGES}）")
    return images


def load_pdf_images() -> tuple[list[Image.Image], Path]:
    pdf_path = resolve_pdf_path()
    print(f"PDFを読み込み: {pdf_path}")
    PAGE_DIR.mkdir(exist_ok=True)
    images = render_pdf_pages(pdf_path)
    for i, image in enumerate(images, start=1):
        image.save(PAGE_DIR / f"page_{i:02d}.png")
    return images, pdf_path


def image_content(model_name: str, image: Image.Image, page_path: Path) -> dict:
    rgb = image.convert("RGB")
    if is_gemma(model_name):
        return {"type": "image", "url": str(page_path.resolve())}
    return {"type": "image", "image": rgb}


def build_messages(
    model_name: str, images: list[Image.Image], prompt: str
) -> list[dict]:
    content: list[dict] = []
    for i, image in enumerate(images, start=1):
        page_path = PAGE_DIR / f"page_{i:02d}.png"
        content.append(image_content(model_name, image, page_path))
        content.append({"type": "text", "text": f"これはPDFの{i}ページ目。"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def apply_chat_inputs(
    processor,
    model_name: str,
    messages: list[dict],
    *,
    enable_thinking: bool | None = None,
):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
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
    elif is_gemma(model_name):
        kwargs["enable_thinking"] = False if enable_thinking is None else enable_thinking
        kwargs["max_soft_tokens"] = GEMMA_MAX_SOFT_TOKENS
    return processor.apply_chat_template(messages, **kwargs)


def move_inputs(inputs, device):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            pass
    moved = {}
    for key, value in dict(inputs).items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def decode_reply(processor, generated, model_name: str, *, enable_thinking: bool | None) -> str:
    if is_muse(model_name):
        raw = processor.decode(generated, skip_special_tokens=False)
        return parse_muse_reply(raw)
    raw = processor.decode(generated, skip_special_tokens=False)
    if is_gemma(model_name) and hasattr(processor, "parse_response"):
        parsed = processor.parse_response(raw, prefix="")
        if isinstance(parsed, dict):
            answer = parsed.get("answer") or parsed.get("content")
            if answer:
                return str(answer).strip()
        return processor.decode(generated, skip_special_tokens=True).strip()
    text = processor.decode(generated, skip_special_tokens=True)
    if is_qwen38(model_name) and enable_thinking:
        return strip_qwen_thinking(text)
    return text.strip()


def load_vlm(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_cls = (
        AutoModelForMultimodalLM
        if (is_muse(model_name) or is_gemma(model_name))
        else AutoModelForImageTextToText
    )
    model = model_cls.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return processor, model


def release_gpu(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_reply(
    model,
    processor,
    model_name: str,
    messages: list[dict],
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, int, float]:
    inputs = apply_chat_inputs(
        processor, model_name, messages, enable_thinking=enable_thinking
    )
    device = next(model.parameters()).device
    inputs = move_inputs(inputs, device)
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
    reply = decode_reply(
        processor, generated, model_name, enable_thinking=enable_thinking
    )
    del inputs, outputs
    return reply, num_tokens, elapsed


def format_run_header(model_name: str, *, thinking: bool | None) -> str:
    short_name = model_name.rsplit("/", 1)[-1]
    if thinking is True:
        mode = "thinking-on"
    elif thinking is False:
        mode = "thinking-off"
    else:
        mode = "default"
    return (
        f"# {short_name}（{mode}）\n\n"
        f"出典: {PDF_URL}\n\n"
    )


def save_model_answers(
    output_path: Path,
    model_name: str,
    sections: list[dict],
    *,
    thinking: bool | None,
) -> None:
    parts = [format_run_header(model_name, thinking=thinking)]
    log_rows = ["| 問 | トークン数 | 推論時間 | 速度 |", "|---|---:|---:|---:|"]
    for section in sections:
        question = section["question"]
        markdown = extract_markdown(section["reply"])
        elapsed = section["elapsed"]
        num_tokens = section["num_tokens"]
        speed = num_tokens / elapsed if elapsed else 0.0
        parts.append(f"## {question['title']}\n\n{markdown.rstrip()}\n")
        log_rows.append(
            f"| {question['title']} | {num_tokens} | {elapsed:.2f}s | {speed:.1f} t/s |"
        )
        print(f"  {question['title']}: {num_tokens} tok, {elapsed:.2f}s, {speed:.1f} t/s")

    parts.append("## 推論ログ\n\n" + "\n".join(log_rows) + "\n")
    if torch.cuda.is_available():
        parts.append(f"\n最大 GPU メモリ使用量: {memory_gb():.2f} GiB\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"生成Markdown: {output_path.resolve()}")
    print()


def run_questions(
    model,
    processor,
    model_name: str,
    images: list[Image.Image],
    *,
    thinking: bool | None,
    label: str = "",
) -> None:
    output_path = output_path_for(model_name, thinking=thinking)
    if label:
        print(f"出力 ({label}): {output_path}")
    else:
        print(f"出力: {output_path}")

    sections = []
    for question in QUESTIONS:
        print(f"-- {question['title']}")
        prompt = build_question_prompt(question)
        messages = build_messages(model_name, images, prompt)
        reply, num_tokens, elapsed = generate_reply(
            model,
            processor,
            model_name,
            messages,
            enable_thinking=thinking,
        )
        print(extract_markdown(reply), end="")
        print()
        sections.append(
            {
                "question": question,
                "reply": reply,
                "num_tokens": num_tokens,
                "elapsed": elapsed,
            }
        )
    save_model_answers(output_path, model_name, sections, thinking=thinking)


def run_one(model_name: str, images: list[Image.Image]) -> None:
    print("=" * 60)
    print(f"モデル: {model_name}")
    print("=" * 60)

    processor, model = load_vlm(model_name)

    if is_qwen38(model_name):
        run_questions(
            model, processor, model_name, images, thinking=False, label="thinking-off"
        )
        run_questions(
            model, processor, model_name, images, thinking=True, label="thinking-on"
        )
    else:
        run_questions(model, processor, model_name, images, thinking=None)

    release_gpu(model, processor)


if __name__ == "__main__":
    images, pdf_path = load_pdf_images()
    print(f"PDF: {pdf_path}（{len(images)}ページ, DPI={RENDER_DPI}）")
    print(f"ページ画像: {PAGE_DIR}")
    print()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("CUDA を使用して推論を実行します。")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise SystemExit("CUDA が使えないため推論を中止します。PDFとページ画像は作成済みです。")
    for model_name in MODEL_NAMES:
        run_one(model_name, images)
