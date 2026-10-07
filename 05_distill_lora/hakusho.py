"""HakushoBench の図表 VQA を、短い読み取り型のプロンプトにする."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from io import BytesIO
from pathlib import Path


class _UnusedAudioDecoder:
    pass


class _UnusedVideoDecoder:
    pass


def _stub_module(name: str, is_package: bool = False):
    """
    find_spec が落ちない空モジュールを返す.

    Args:
        name (str): `sys.modules` に置く名前.
        is_package (bool): 子モジュールを持つなら True. 既定は False.

    Returns:
        types.ModuleType: `__spec__` を付けた空モジュール.

    Example:
        package = _stub_module("torchcodec", is_package=True)

    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name,
        loader=None,
        is_package=is_package,
    )
    if is_package:
        module.__path__ = []
    return module


def _block_broken_torchcodec() -> None:
    """
    Unsloth が datasets 経由で torchcodec の .so を読むのを止める.

    HakushoBench の図表は parquet の画像だけなので、音声・動画デコーダは使わない.

    Returns:
        None: `sys.modules` に空の torchcodec を先に置く.

    Example:
        _block_broken_torchcodec()

    """
    if "torchcodec" in sys.modules:
        return
    decoders = _stub_module("torchcodec.decoders")
    decoders.AudioDecoder = _UnusedAudioDecoder
    decoders.VideoDecoder = _UnusedVideoDecoder
    package = _stub_module("torchcodec", is_package=True)
    package.decoders = decoders
    package.encoders = _stub_module("torchcodec.encoders")
    package.samplers = _stub_module("torchcodec.samplers")
    package.transforms = _stub_module("torchcodec.transforms")
    sys.modules["torchcodec"] = package
    sys.modules["torchcodec.decoders"] = decoders
    sys.modules["torchcodec.encoders"] = package.encoders
    sys.modules["torchcodec.samplers"] = package.samplers
    sys.modules["torchcodec.transforms"] = package.transforms


_block_broken_torchcodec()

from datasets import Dataset, load_dataset
from PIL import Image

from readout import ChartReadout


class HakushoSet:
    """
    クローンした HakushoBench を読み、図表と質問のプロンプトにする.

    画像は Hugging Face には無く、GitLab の parquet にある.
    公式の読み方は `load_dataset("hakushobench/data", split="test")`.

    Attributes:
        CHART_TYPES (tuple[str, ...]): 残すグラフの image_type.
        LONG_SIDE (int): 図表の長辺の上限ピクセル. 768.
        EVAL_RATIO (float): テストに回す割合. 0.05.
        SEED (int): 分割と LoRA の乱数. 3407.
        repo_dir (Path): `git clone` した hakushobench のルート.

    Methods:
        load_rows: test parquet を Dataset として読む.
        filter_charts: 棒・折れ線・円などのグラフだけ残す.
        split_dataset: 学習用とテスト用に分ける.
        build_prompts: 各行を prompt・image・answer にする.
        load_chart: 1 行の図表を RGB の PIL にする.

    Example:
        hakusho = HakushoSet(repo_dir="./hakushobench")
        rows = hakusho.filter_charts(hakusho.load_rows())
        train_dataset, test_dataset = hakusho.split_dataset(
            rows, eval_ratio=HakushoSet.EVAL_RATIO, seed=HakushoSet.SEED
        )

    """

    CHART_TYPES = ("Area", "Bar", "Bubble", "Line", "Pie", "Scatter")
    LONG_SIDE = 768
    EVAL_RATIO = 0.05
    SEED = 3407

    def __init__(self, repo_dir: str | Path):
        self.repo_dir = Path(repo_dir)

    def load_rows(self) -> Dataset:
        """
        `data/test-*.parquet` を test split として読む.

        Returns:
            Dataset: `question`、`answer`、`image` を持つ 2053 行前後.

        Example:
            rows = HakushoSet(repo_dir="./hakushobench").load_rows()

        """
        data_dir = self.repo_dir / "data"
        parquet_paths = sorted(data_dir.glob("test-*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(
                f"{data_dir} に test-*.parquet がありません."
                " https://gitlab.llm-jp.nii.ac.jp/datasets/hakushobench を clone してください."
            )
        return load_dataset(
            "parquet",
            data_files={"test": [str(path) for path in parquet_paths]},
            split="test",
        )

    def filter_charts(self, rows: Dataset) -> Dataset:
        """
        グラフ系の図表だけを残す.

        残すのは面、棒、バブル、折れ線、円、散布図.
        表、地図、インフォグラフィック、ダッシュボード、その他は除く.
        SFT と蒸留は表や地図を使わず、このあとに `split_dataset` する.

        Args:
            rows (Dataset): `load_rows` が返した HakushoBench.

        Returns:
            Dataset: `image_type` がグラフ系の行だけ.

        Example:
            rows = hakusho.filter_charts(hakusho.load_rows())

        """
        return rows.filter(lambda row: row["image_type"] in HakushoSet.CHART_TYPES)

    def split_dataset(
        self, rows: Dataset, eval_ratio: float, seed: int
    ) -> tuple[Dataset, Dataset]:
        """
        グラフ系の行を、学習用とテスト用に分ける.

        教師データの作成、SFT、採点は同じ割合と seed でこの分割を使う.
        テスト用は学習に入れない.

        Args:
            rows (Dataset): `filter_charts` が残したグラフ系の行.
            eval_ratio (float): テストに回す割合. 0 なら全件を学習に使う.
            seed (int): 分割の乱数.

        Returns:
            tuple[Dataset, Dataset]: `train_dataset` と `test_dataset`. 割合が 0 のとき、テストは空.

        Example:
            train_dataset, test_dataset = hakusho.split_dataset(
                rows, eval_ratio=HakushoSet.EVAL_RATIO, seed=HakushoSet.SEED
            )

        """
        if eval_ratio <= 0:
            return rows, rows.select([])
        split_rows = rows.train_test_split(test_size=eval_ratio, seed=seed)
        return split_rows["train"], split_rows["test"]

    def build_prompts(self, rows: Dataset) -> list[dict]:
        """
        図表と質問を user の prompt にし、正解は採点用の answer として残す.

        空の質問や回答は捨てる. 画像は RGB の PIL にする.
        思考タグは使わない. 根拠には読み取った項目・数値・単位と比較を書く. 正解の文面は教師にしない.
        順位の問いだけ、同じ単位の値を問いの順に書く.
        国名の順位ではその他を国とみなさない.

        Args:
            rows (Dataset): `load_rows` が返した HakushoBench.

        Returns:
            list[dict]: `prompt`、`image`、`answer` を持つ行.

        Example:
            samples = HakushoSet(repo_dir="./hakushobench").build_prompts(rows)

        """
        readout = ChartReadout()
        samples = []
        for row in rows:
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            if not question or not answer:
                continue
            chart = self.load_chart(row)
            samples.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": readout.prompt(question)},
                            ],
                        }
                    ],
                    "image": chart,
                    "answer": answer,
                    "original_id": str(row["original_id"]),
                }
            )
        return samples

    def load_chart(self, row: dict) -> Image.Image:
        """
        1 行の図表を RGB の PIL 画像にし、長辺が LONG_SIDE を超えていれば縮小する.

        Args:
            row (dict): `image` または `image_path` を持つ 1 行.

        Returns:
            PIL.Image.Image: RGB で、長辺を LONG_SIDE 以内にした図表.

        Example:
            chart = hakusho.load_chart(rows[0])

        """
        image = row.get("image")
        if isinstance(image, Image.Image):
            chart = image.convert("RGB")
        elif isinstance(image, dict) and image.get("bytes"):
            chart = Image.open(BytesIO(image["bytes"])).convert("RGB")
        else:
            chart = Image.open(self._image_path(row, image)).convert("RGB")
        return self._fit_long_side(chart)

    def _fit_long_side(self, chart: Image.Image) -> Image.Image:
        """
        長辺が LONG_SIDE を超える図表を、縦横比を保って縮小する.

        Args:
            chart (PIL.Image.Image): RGB の図表.

        Returns:
            PIL.Image.Image: 長辺が LONG_SIDE 以内の図表. すでに小さければそのまま.

        Example:
            fitted = hakusho._fit_long_side(chart)

        """
        width, height = chart.size
        longest = max(width, height)
        if longest <= HakushoSet.LONG_SIDE:
            return chart
        scale = HakushoSet.LONG_SIDE / longest
        return chart.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    def _image_path(self, row: dict, image) -> Path:
        """
        行に書かれた画像パスを、実在するファイルパスに解決する.

        Args:
            row (dict): `image_path` を持ちうる 1 行.
            image (object): `path` を持ちうる image 列の値.

        Returns:
            Path: 開ける画像ファイル.

        Example:
            path = hakusho._image_path(rows[0], rows[0]["image"])

        """
        raw_path = row.get("image_path") or ""
        if not raw_path and isinstance(image, dict):
            raw_path = image.get("path") or ""
        if not raw_path:
            raise FileNotFoundError("image も image_path もありません.")
        image_path = Path(raw_path)
        if image_path.is_file():
            return image_path
        rooted = self.repo_dir / raw_path
        if rooted.is_file():
            return rooted
        raise FileNotFoundError(f"図表が見つかりません: {raw_path}")
