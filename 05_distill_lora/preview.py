"""HakushoBench の図表の種類と、質問・正解の例を確認する.

使い方:
    python preview.py --hakusho /path/to/hakushobench
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset

from hakusho import HakushoSet


class HakushoPreview:
    """
    公式の読み方で HakushoBench を開き、種類の内訳と数件の質問・正解を出す.

    サンプル図表は PNG で残し、同じ名前のテキストに質問と正解を書く.

    Attributes:
        repo_dir (Path): `git clone` した hakushobench のルート.
        out_dir (Path): サンプル画像と質問文の保存先.
        n (int): 保存する件数.
        hakusho (HakushoSet): 図表を PIL にする読み手.

    Methods:
        load: `data` ディレクトリを test split として読む.
        show: 内訳を表示し、種類がばらけるようにサンプルを保存する.

    Example:
        preview = HakushoPreview(repo_dir="./hakushobench", out_dir="./preview_samples", n=8)
        preview.show(preview.load())

    """

    def __init__(self, repo_dir: str | Path, out_dir: str | Path, n: int = 8):
        self.repo_dir = Path(repo_dir)
        self.out_dir = Path(out_dir)
        self.n = n
        self.hakusho = HakushoSet(repo_dir=self.repo_dir)

    def load(self) -> Dataset:
        """
        clone 先の `data` を、公式と同じ `load_dataset(..., split="test")` で読む.

        Returns:
            Dataset: `image`、`question`、`answer` などを持つ test split.

        Example:
            rows = HakushoPreview(
                repo_dir="./hakushobench",
                out_dir="./preview_samples",
            ).load()

        """
        data_dir = self.repo_dir / "data"
        return load_dataset(str(data_dir), split="test")

    def show(self, rows: Dataset) -> None:
        """
        行数と種類の内訳を表示し、図表の種類が分かれるサンプルを保存する.

        Args:
            rows (Dataset): `load` が返した test split.

        Returns:
            None: 標準出力と out_dir への保存が副作用.

        Example:
            preview.show(rows)

        """
        print(rows)
        for column in ("image_type", "category", "question_type"):
            print(f"\n[{column}]")
            labels = [self._label(value) for value in rows[column]]
            for name, count in Counter(labels).most_common():
                print(f"  {count:4d}  {name}")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for index in self._sample_indices(rows):
            row = rows[index]
            chart = self.hakusho.load_chart(row)
            image_type = self._file_stem(str(row["image_type"]))
            stem = f"{index:04d}_{image_type}"
            image_path = self.out_dir / f"{stem}.png"
            chart.save(image_path)
            note_path = self.out_dir / f"{stem}.txt"
            note_path.write_text(
                "\n".join(
                    [
                        f"original_id: {row['original_id']}",
                        f"category: {row['category']}",
                        f"image_type: {row['image_type']}",
                        f"question_type: {self._label(row['question_type'])}",
                        f"size: {chart.size[0]}x{chart.size[1]}",
                        "",
                        "質問:",
                        str(row["question"]).strip(),
                        "",
                        "正解:",
                        str(row["answer"]).strip(),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            print(f"\n--- {stem} {chart.size[0]}x{chart.size[1]} ---")
            print(note_path.read_text(encoding="utf-8"))
        print("saved", self.out_dir.resolve())

    def _label(self, value) -> str:
        """
        列の値を、件数集計に使える1行の文字列にする.

        `question_type` は辞書で入っているため、そのまま Counter に渡せない.

        Args:
            value (object): 列の1件。文字列、辞書、リストのいずれか.

        Returns:
            str: 表示と集計に使うラベル.

        Example:
            label = preview._label({"task": "read", "skill": "compare"})

        """
        if isinstance(value, dict):
            parts = [f"{key}={self._label(item)}" for key, item in value.items()]
            return ", ".join(parts)
        if isinstance(value, list):
            return " | ".join(self._label(item) for item in value)
        return str(value)

    def _sample_indices(self, rows: Dataset) -> list[int]:
        """
        図表の種類が先に一通り出るよう、サンプル行の番号を返す.

        Args:
            rows (Dataset): test split.

        Returns:
            list[int]: 最大 `n` 件の行番号.

        Example:
            indices = preview._sample_indices(rows)

        """
        by_type: dict[str, list[int]] = {}
        for index, image_type in enumerate(rows["image_type"]):
            by_type.setdefault(str(image_type), []).append(index)
        picked = [indices[0] for indices in by_type.values()]
        picked = picked[: self.n]
        seen = set(picked)
        for index in range(len(rows)):
            if len(picked) >= self.n:
                break
            if index not in seen:
                picked.append(index)
        return picked

    def _file_stem(self, image_type: str) -> str:
        """
        図表種別を、ファイル名に使える短い文字列にする.

        Args:
            image_type (str): データセットの image_type.

        Returns:
            str: 英数字とアンダースコアだけの名前.

        Example:
            stem = preview._file_stem("折れ線グラフ")

        """
        chars = [ch if ch.isalnum() else "_" for ch in image_type.strip()]
        stem = "".join(chars).strip("_")
        return stem[:40] or "image"


def main():
    parser = argparse.ArgumentParser(
        description="HakushoBench の図表・質問・正解を数件確認する."
    )
    parser.add_argument(
        "--hakusho",
        required=True,
        help="clone した hakushobench リポジトリのパス.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "preview_samples"),
        help="サンプル画像と質問文の保存先.",
    )
    parser.add_argument("--n", type=int, default=8, help="保存する件数.")
    args = parser.parse_args()

    preview = HakushoPreview(
        repo_dir=args.hakusho,
        out_dir=args.out_dir,
        n=args.n,
    )
    preview.show(preview.load())


if __name__ == "__main__":
    main()
