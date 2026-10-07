"""経産省 PDF で、思考オフの Gemma 4 が前回外した問だけを確認する.

画像は pdf_compare.py と同じ（先頭 12 ページ、200dpi、思考オフ）.
文面はベースも SFT も HakushoBench と同じ ChartReadout.
15 問の表埋めは使わない. p15 だけを同じプロンプトで投げる.

使い方:
    python pdf_miss.py
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

from PIL import Image

from readout import ChartReadout

try:
    import pymupdf
except ImportError:
    pymupdf = None

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MetiMiss:
    """
    1-1-1.pdf のうち、思考オフで前回外した問だけを Gemma 4 の vLLM に送る.

    前回の不正解は `p15`（第1-1-1-10図 タイの4位の相手国）だけ.
    画像は `pdf_compare.py` と同じく、先頭から最大 12 ページを 200dpi で渡す.
    文面はベースも SFT も `ChartReadout.prompt`. 他の問いを混ぜない.

    Attributes:
        pdf_path (Path): 通商白書の PDF.
        model_name (str): vLLM に載っているベースモデル名.
        sft_name (str): SFT LoRA の名前.
        sft_dir (Path): SFT のアダプタ.
        output_dir (Path): jsonl と Markdown の保存先.
        max_new_tokens (int): 生成トークンの上限.
        vllm_url (str): Gemma 4 用 vLLM の OpenAI 互換 URL.
        readout (ChartReadout): プロンプトと正解判定の規則.

    Methods:
        served_ids: 載っているモデル名を返す.
        require_lora: アダプタと vLLM 上の名前が揃っているかを確認する.
        pages: 先頭ページを画像の列にして返す.
        reply: 1 問を思考オフで生成する.
        write: ベースと SFT を同時に書いて Markdown にする.

    Example:
        miss = MetiMiss(pdf_path="./1-1-1.pdf")
        miss.write([("before", miss.model_name), ("sft", miss.sft_name)])

    """

    misses = (
        {
            "id": "p15",
            "figure": "第1-1-1-10図",
            "question": "第1-1-1-10図 タイの4位の相手国",
            "gold": "バーレーン",
            "accept": (),
            "prev": "カタール",
        },
    )
    render_dpi = 200
    max_pages = 12

    def __init__(
        self,
        pdf_path: str | Path,
        model_name: str = "google/gemma-4-31B-it",
        sft_name: str = "sft",
        sft_dir: str | Path = "outputs/gemma4_hakusho_sft",
        output_dir: str | Path = "outputs/pdf_miss",
        max_new_tokens: int = 1536,
        vllm_url: str = "http://127.0.0.1:8000",
    ):
        self.pdf_path = Path(pdf_path)
        self.model_name = model_name
        self.sft_name = sft_name
        self.sft_dir = Path(sft_dir)
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.vllm_url = vllm_url
        self.readout = ChartReadout()

    def served_ids(self) -> set[str]:
        """
        vLLM に載っているモデル名を返す.

        Returns:
            set[str]: `/v1/models` の id.

        Example:
            names = miss.served_ids()

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
            miss.require_lora("SFT", miss.sft_dir, miss.sft_name, served)

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

    def pages(self) -> list[Image.Image]:
        """
        PDF の先頭から最大 max_pages 枚を、200dpi の RGB 画像にして返す.

        `pdf_compare.py` と同じ枚数・解像度. 縮小しない.

        Returns:
            list[PIL.Image.Image]: 1 ページ目からの画像.

        Example:
            charts = miss.pages()

        """
        if pymupdf is None:
            raise SystemExit("PDF のページ化には pymupdf が必要です: uv add pymupdf")
        if not self.pdf_path.is_file():
            raise SystemExit(f"PDF がありません: {self.pdf_path}")
        doc = pymupdf.open(self.pdf_path)
        zoom = self.render_dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        limit = min(doc.page_count, self.max_pages)
        charts = []
        for index in range(limit):
            pix = doc[index].get_pixmap(matrix=matrix, alpha=False)
            charts.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        if not charts:
            raise SystemExit(f"PDF からページを読めません: {self.pdf_path}")
        return charts

    def reply(
        self,
        item: dict,
        charts: list[Image.Image],
        model_id: str,
    ) -> dict:
        """
        1 問を思考オフで読ませ、根拠・答え・正解かどうかを返す.

        ベースも SFT も `ChartReadout.prompt`. p15 以外の問いは入れない.

        Args:
            item (dict): `id`、`question`、`gold`、`accept`、`prev`、`figure`.
            charts (list[PIL.Image.Image]): PDF の先頭ページ.
            model_id (str): ベースなら `model_name`. LoRA なら `sft_name`.

        Returns:
            dict: 問い、正解、前回の答え、生成、`hit`.

        Example:
            record = miss.reply(MetiMiss.misses[0], charts, miss.sft_name)

        """
        question = str(item["question"])
        gold = str(item["gold"])
        prompt = self.readout.prompt(question)
        raw = self._vllm_reply(charts, prompt, model_id)
        lines = self.readout.visible_lines(raw)
        index = self.readout._answer_index(lines)
        if index is None:
            grounds, answer_line = "\n".join(lines), ""
        else:
            grounds, answer_line = "\n".join(lines[:index]), lines[index]
        body = self.readout.answer_body(raw)
        keys = [self.readout.norm(gold)]
        keys.extend(self.readout.norm(str(alias)) for alias in item.get("accept", ()))
        return {
            "id": item["id"],
            "figure": item["figure"],
            "question": question,
            "gold": gold,
            "prev": item["prev"],
            "raw": raw,
            "grounds": grounds,
            "answer_line": answer_line,
            "hit": body is not None and body in keys,
        }

    def _vllm_reply(self, charts: list[Image.Image], prompt: str, model_id: str) -> str:
        """
        全ページ画像とユーザー文を vLLM に送り、本文を返す.

        Args:
            charts (list[PIL.Image.Image]): RGB のページ.
            prompt (str): ChartReadout のユーザー文.
            model_id (str): リクエストの `model`.

        Returns:
            str: 生成本文.

        Example:
            raw = miss._vllm_reply(charts, prompt, "sft")

        """
        content = []
        for index, chart in enumerate(charts, start=1):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._png_data_url(chart)},
                }
            )
            content.append({"type": "text", "text": f"これはPDFの{index}ページ目。"})
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
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

    def _png_data_url(self, chart: Image.Image) -> str:
        """
        ページを PNG の data URL にして返す.

        Args:
            chart (PIL.Image.Image): RGB のページ.

        Returns:
            str: `data:image/png;base64,...` の URL.

        Example:
            url = miss._png_data_url(chart)

        """
        buffer = BytesIO()
        chart.save(buffer, format="PNG")
        encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def write(self, sides: list[tuple[str, str]]) -> None:
        """
        前回外した問を、指定したモデルで同時に生成し、jsonl と Markdown に書く.

        Args:
            sides (list[tuple[str, str]]): (jsonl の stem, vLLM の model id) の列.

        Returns:
            None: output_dir に jsonl と pdf_miss.md を書く.

        Example:
            miss.write([("before", miss.model_name), ("sft", miss.sft_name)])

        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        charts = self.pages()
        print(f"pages {len(charts)} dpi {self.render_dpi}", flush=True)
        by_side = {stem: [] for stem, _model_id in sides}
        for item in self.misses:
            print(f"{item['id']} {item['figure']}", flush=True)
            with ThreadPoolExecutor(max_workers=len(sides)) as pool:
                futures = {
                    stem: pool.submit(
                        self.reply,
                        item,
                        charts,
                        model_id,
                    )
                    for stem, model_id in sides
                }
                for stem, future in futures.items():
                    record = future.result()
                    record["pages"] = len(charts)
                    by_side[stem].append(record)
                    mark = "正解" if record["hit"] else "不正解"
                    print(f"  {stem} {mark} {record['answer_line']}", flush=True)
        for stem, rows in by_side.items():
            path = self.output_dir / f"{stem}.jsonl"
            with path.open("w", encoding="utf-8") as traces:
                for record in rows:
                    traces.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._report(by_side, sides)

    def _report(
        self,
        by_side: dict[str, list[dict]],
        sides: list[tuple[str, str]],
    ) -> None:
        """
        各側の jsonl から、前回外した問の Markdown を書く.

        Args:
            by_side (dict[str, list[dict]]): stem から生成レコードの列.
            sides (list[tuple[str, str]]): 見出し順に使う stem と model id.

        Returns:
            None: pdf_miss.md を書く.

        Example:
            miss._report(by_side, [("before", miss.model_name)])

        """
        labels = {"before": "ベース", "sft": "SFT", "after": "SFT"}
        lines = [
            "# 経産省 PDF の前回不正解",
            "",
            "思考はオフ. ベースも SFT も同じプロンプト（降順に書く、その他は国とみなさない）.",
            "問いは p15 だけ. 15 問の表埋めは使わない.",
            f"PDF: `{self.pdf_path}`（先頭 {self.max_pages} ページ、{self.render_dpi}dpi）",
            "",
        ]
        for item in self.misses:
            lines.append(f"## {item['id']} {item['question']}")
            lines.append("")
            lines.append(f"- 正解: {item['gold']}")
            lines.append(f"- 前回: {item['prev']}")
            lines.append("")
            for stem, _model_id in sides:
                record = next(row for row in by_side[stem] if row["id"] == item["id"])
                label = labels.get(stem, stem)
                mark = "正解" if record["hit"] else "不正解"
                answer = record["answer_line"].strip() or "（答えの行なし）"
                grounds = record["grounds"].strip() or "（根拠なし）"
                lines.extend(
                    [
                        f"### {label}: {mark}",
                        "",
                        "根拠",
                        "",
                        f"```\n{grounds.replace('```', "'''")}\n```",
                        "",
                        "回答",
                        "",
                        f"```\n{answer.replace('```', "'''")}\n```",
                        "",
                    ]
                )
        report_path = self.output_dir / "pdf_miss.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("report", report_path.resolve())


def find_pdf(given: str) -> Path:
    """
    引数か、このフォルダの 1-1-1.pdf を返す.

    Args:
        given (str): `--pdf` の値. 空ならこのスクリプトと同じフォルダを見る.

    Returns:
        Path: 存在する PDF.

    Example:
        pdf_path = find_pdf("")

    """
    here = Path(__file__).resolve().parent
    if given:
        path = Path(given)
        if path.is_file():
            return path
        raise SystemExit(f"PDF がありません: {path}")
    for candidate in (here / "1-1-1.pdf", Path.cwd() / "1-1-1.pdf"):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"1-1-1.pdf が見つかりません. {here} に置くか --pdf でパスを渡してください."
    )


def main():
    parser = argparse.ArgumentParser(
        description="経産省 PDF で、思考オフが前回外した問だけを確認する."
    )
    parser.add_argument(
        "--pdf", default="", help="1-1-1.pdf のパス. 空ならこのフォルダ."
    )
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--sft-name", default="sft")
    parser.add_argument(
        "--sft-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "gemma4_hakusho_sft"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "pdf_miss"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--side",
        choices=("before", "sft", "both"),
        default="both",
        help="before はベース、sft は SFT、both は両方.",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    miss = MetiMiss(
        pdf_path=find_pdf(args.pdf),
        model_name=args.model,
        sft_name=args.sft_name,
        sft_dir=args.sft_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        vllm_url=args.vllm_url,
    )
    try:
        served = miss.served_ids()
    except RuntimeError as error:
        raise SystemExit(
            f"{error}\n先に bash serve_vllm.sh gemma を起動してください."
        ) from error
    print("vllm", ", ".join(sorted(served)))
    print("pdf", miss.pdf_path)
    if args.side in ("sft", "both"):
        miss.require_lora("SFT", miss.sft_dir, miss.sft_name, served)
    sides = []
    if args.side in ("before", "both"):
        sides.append(("before", miss.model_name))
    if args.side in ("sft", "both"):
        sides.append(("sft", miss.sft_name))
    miss.write(sides)


if __name__ == "__main__":
    main()
