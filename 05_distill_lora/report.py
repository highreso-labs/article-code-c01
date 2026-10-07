"""学習規模と所要時間を、ターミナルと HTML に残す."""

from __future__ import annotations

import html
from pathlib import Path


class TrainReport:
    """
    学習条件と経過時間を表にし、ターミナルと HTML の両方へ出す.

    Attributes:
        title (str): HTML の見出し.
        rows (list[tuple[str, str]]): ラベルと値の組.
        counts (list[tuple[str, list[tuple[str, int]]]]): 内訳表の見出しと件数.

    Methods:
        add: 1 行を足す.
        add_counts: 種類ごとの件数表を足す.
        clock: 秒を時間表示にする.
        show: ターミナルに出す.
        save: HTML を書く.

    Example:
        report = TrainReport(title="HakushoBench SFT")
        report.add("学習件数", 476)
        report.show()
        report.save(path="sft_report.html")

    """

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, str]] = []
        self.counts: list[tuple[str, list[tuple[str, int]]]] = []

    def add(self, label: str, value) -> None:
        """
        結果表に 1 行足す.

        Args:
            label (str): 行の名前.
            value (object): 表示する値. 文字列にする.

        Returns:
            None: 行を rows に足す.

        Example:
            report.add("学習件数", 476)

        """
        self.rows.append((label, str(value)))

    def add_counts(self, title: str, pairs: list[tuple[str, int]]) -> None:
        """
        種類と件数の内訳表を足す.

        Args:
            title (str): 内訳の見出し.
            pairs (list[tuple[str, int]]): 名前と件数. 件数の多い順.

        Returns:
            None: 内訳を counts に足す.

        Example:
            report.add_counts("image_type", [("Table", 10)])

        """
        self.counts.append((title, pairs))

    def clock(self, seconds: float) -> str:
        """
        秒を、時間分秒と時間数の文字列にする.

        Args:
            seconds (float): 経過秒.

        Returns:
            str: `1時間02分03秒（1.03 時間）` の形.

        Example:
            text = report.clock(3723)

        """
        whole = int(round(seconds))
        hours, remainder = divmod(whole, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}時間{minutes:02d}分{secs:02d}秒（{seconds / 3600:.2f} 時間）"

    def show(self) -> None:
        """
        結果表と内訳をターミナルに出す.

        Returns:
            None: 標準出力への表示が副作用.

        Example:
            report.show()

        """
        print("\n======== 学習結果 ========")
        width = max(len(label) for label, _ in self.rows)
        for label, value in self.rows:
            print(f"{label:<{width}}  {value}")
        for title, pairs in self.counts:
            print(f"\n[{title}]")
            for name, count in pairs:
                print(f"  {count:4d}  {name}")

    def save(self, path: str | Path) -> Path:
        """
        結果表と内訳を HTML として保存する.

        Args:
            path (str | Path): 書き出す HTML のパス.

        Returns:
            Path: 保存したファイル.

        Example:
            report.save(path="sft_report.html")

        """
        html_path = Path(path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        body = [
            "<!DOCTYPE html>",
            '<html lang="ja">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{html.escape(self.title)}</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 2rem; color: #1f2937; background: #fff; }",
            "table { border-collapse: collapse; margin: 1rem 0 2rem; }",
            "td, th { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{html.escape(self.title)}</h1>",
            "<table>",
        ]
        for label, value in self.rows:
            body.append(
                "<tr>"
                f"<th>{html.escape(label)}</th>"
                f"<td>{html.escape(value)}</td>"
                "</tr>"
            )
        body.append("</table>")
        for title, pairs in self.counts:
            body.append(f"<h2>{html.escape(title)}</h2>")
            body.append("<table><tr><th>種類</th><th>件数</th></tr>")
            for name, count in pairs:
                body.append(
                    "<tr>"
                    f"<td>{html.escape(str(name))}</td>"
                    f"<td>{count}</td>"
                    "</tr>"
                )
            body.append("</table>")
        body.append("</body></html>")
        html_path.write_text("\n".join(body), encoding="utf-8")
        return html_path
