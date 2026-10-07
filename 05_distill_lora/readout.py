"""図表の読み取り型. 根拠のあとに、最後の行を「答え:」で始める."""

from __future__ import annotations

import re
import unicodedata


class ChartReadout:
    """
    図表 VQA の短い読み取り型を、プロンプトと採点で同じ規則にする.

    思考オンの教師は思考タグの中で考える. 残すのはタグの外の根拠と「答え:」だけ.
    順位の問いでは、同じ単位の値を問いの順に書いた文だけ残す.
    相手国・国名の順位では、その他を順位や答えに入れた文は残さない.
    思考タグは採点にも教師データの保存にも使わない.

    Attributes:
        mark (str): 答えの行の先頭. 「答え:」.
        max_grounds (int): 答えの行より前に許す空でない行数. 12.
        max_chars (int): 空白を除いた文字数の上限. 400.

    Methods:
        prompt: 質問を、この型を求めるユーザー文にする.
        norm: 比較用に表記を揃える.
        visible: 思考タグより後の本文を返す.
        answer_body: 「答え:」の右側を正規化して返す.
        grounds: 答えの行より前の根拠を返す.
        too_long: 行数か文字数が上限を超えているか返す.
        answer_closed: 答えの行が改行まで出ているか返す.
        judge: 教師の生成を、SFT に残す文か捨てる理由にする.

    Example:
        readout = ChartReadout()
        prompt = readout.prompt(question="4位の国はどこか。")
        completion, reason = readout.judge(
            text="バーレーン\\n答え: バーレーン\\n",
            gold="バーレーン",
            question="4位の国はどこか。",
        )

    """

    mark = "答え:"
    max_grounds = 12
    max_chars = 400
    _rank_q = re.compile(r"位|番目|順位")
    _country_rank_q = re.compile(
        r"相手国|(?:位|番目|順位).{0,8}国|国.{0,8}(?:位|番目|順位)"
    )
    _ascend_q = re.compile(r"小さい|最小|下位|少ない")
    _other = re.compile(r"その他")
    _exclude_other = re.compile(
        r"その他.{0,16}(?:除|含めない|含まない|みなさない|外[すし])"
        r"|(?:除|含めない|含まない|みなさない|外[すし]).{0,16}その他"
    )
    _percent = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
    _count = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+)\s*件")

    def prompt(self, question: str) -> str:
        """
        図表の質問を、根拠と「答え:」の1行を求めるユーザー文にする.

        Args:
            question (str): HakushoBench の質問.

        Returns:
            str: 型の指示と質問を連結した文.

        Example:
            text = ChartReadout().prompt(question="4位の国はどこか。")

        """
        return (
            "この図表を読み、答えに必要な根拠を書いてください。"
            "根拠には、読み取った項目・数値・単位と、問いが求める比較を含めてください。"
            + (
                "順位は同じ単位の値を問いの順（大きい順なら降順）に書いてください。"
                if self._rank_q.search(question)
                else ""
            )
            + (
                "相手国や国名の順位では、「その他」は国とみなさないでください。"
                if self._country_rank_q.search(question)
                else ""
            )
            + "最後の行を「答え:」で始め、その右側に問いの短い答えだけを書いてください。\n"
            + question
        )

    def norm(self, text: str) -> str:
        """
        全角・空白・波ダッシュ・ハイフン・囲み括弧・末尾の句点を揃えて、比較用の文字列を返す.

        Args:
            text (str): 正解、またはモデルの出力.

        Returns:
            str: NFKC 正規化し、空白と末尾の句点を除いた文字列.
            `〜` `～` `-` は同じ. 答え全体を囲む `()` は除く.

        Example:
            key = ChartReadout().norm("(35〜39歳)")

        """
        folded = unicodedata.normalize("NFKC", text)
        for mark in ("〜", "～", "~", "―", "–", "—", "−"):
            folded = folded.replace(mark, "-")
        folded = re.sub(r"\s+", "", folded)
        while True:
            stripped = folded.strip("。．.、,")
            if len(stripped) >= 2 and stripped[0] == "(" and stripped[-1] == ")":
                folded = stripped[1:-1]
                continue
            return stripped

    def visible(self, text: str) -> str:
        """
        思考タグより後の本文を返す. タグが閉じるまでは空にする.

        Args:
            text (str): モデル出力. 思考タグを含み得る.

        Returns:
            str: `</think>` より後. 思考の途中なら空文字.

        Example:
            body = ChartReadout().visible("<think>読む</think>\\n答え: 3.4倍")

        """
        for token in (
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<turn|>",
            "<|turn>",
            "<|channel>",
            "<channel|>",
        ):
            text = text.replace(token, "")
        end = text.find("</think>")
        if end >= 0:
            return text[end + len("</think>") :].strip()
        if text.find("<think>") >= 0:
            return ""
        return text.strip()

    def visible_lines(self, text: str) -> list[str]:
        """
        思考を除いた本文を、空でない行のリストにして返す.

        Args:
            text (str): モデル出力.

        Returns:
            list[str]: 前後の空白を除いた行. 空行は含まない.

        Example:
            lines = ChartReadout().visible_lines("項目\\n\\n答え: 3.4倍\\n")

        """
        return [
            line.strip() for line in self.visible(text).splitlines() if line.strip()
        ]

    def _answer_index(self, lines: list[str]) -> int | None:
        """
        「答え:」で始まる最後の行の位置を返す.

        Args:
            lines (list[str]): `visible_lines` の結果.

        Returns:
            int | None: 行の位置. 無ければ None.

        Example:
            index = ChartReadout()._answer_index(["項目", "答え: 3.4倍"])

        """
        for index in range(len(lines) - 1, -1, -1):
            if self.norm(lines[index]).startswith(self.mark):
                return index
        return None

    def answer_body(self, text: str) -> str | None:
        """
        「答え:」の右側を正規化して返す. その行が無ければ None.

        Args:
            text (str): モデル出力.

        Returns:
            str | None: 答えの行の右側. 行が無ければ None.

        Example:
            body = ChartReadout().answer_body("項目\\n答え: (3.4倍)\\n")

        """
        lines = self.visible_lines(text)
        index = self._answer_index(lines)
        if index is None:
            return None
        folded = self.norm(lines[index])
        return self.norm(folded[len(self.mark) :])

    def grounds(self, text: str) -> str:
        """
        答えの行より前の根拠を、改行でつないで返す.

        Args:
            text (str): モデル出力.

        Returns:
            str: 根拠の行. 答えの行が無ければ本文の行全部.

        Example:
            grounds = ChartReadout().grounds("バーレーン\\n答え: バーレーン\\n")

        """
        lines = self.visible_lines(text)
        index = self._answer_index(lines)
        if index is None:
            return "\n".join(lines)
        return "\n".join(lines[:index])

    def too_long(self, text: str) -> bool:
        """
        根拠が12行を超えるか、答えの行の後に文があるか、400字を超えるかを返す.

        Args:
            text (str): モデル出力.

        Returns:
            bool: 上限を超えていれば True.

        Example:
            long = ChartReadout().too_long("a\\n" * 14 + "答え: 5回\\n")

        """
        lines = self.visible_lines(text)
        index = self._answer_index(lines)
        grounds = lines if index is None else lines[:index]
        if len(grounds) > self.max_grounds:
            return True
        if index is not None and index != len(lines) - 1:
            return True
        return len(self.norm("\n".join(lines))) > self.max_chars

    def answer_closed(self, text: str) -> bool:
        """
        生成中の文に、「答え:」の行が改行まで出ているかを返す.

        全角コロンも同じ印として見る. 思考の外かどうかは見ない.

        Args:
            text (str): そこまでの生成をデコードした文.

        Returns:
            bool: 答えの印の後に改行があれば True.

        Example:
            closed = ChartReadout().answer_closed("項目\\n答え: 3.4倍\\n")

        """
        for mark in (self.mark, "答え："):
            at = text.rfind(mark)
            if at >= 0 and "\n" in text[at:]:
                return True
        return False

    def _value_groups(self, grounds: str) -> list[list[float]]:
        """
        根拠から、同じ単位の数値の列を返す.

        百分率は合計の 100 を除く. 件はカンマを除いて読む.
        2つ以上ある単位だけを残す.

        Args:
            grounds (str): 答えの行より前の本文.

        Returns:
            list[list[float]]: 単位ごとの出現順の数値.

        Example:
            groups = ChartReadout()._value_groups("38.5% 26.4% 11.5% 9.1%")

        """
        percents = [
            float(match.group(1))
            for match in self._percent.finditer(grounds)
            if float(match.group(1)) != 100.0
        ]
        counts = [
            float(match.group(1).replace(",", ""))
            for match in self._count.finditer(grounds)
        ]
        groups = []
        if len(percents) >= 2:
            groups.append(percents)
        if len(counts) >= 2:
            groups.append(counts)
        return groups

    def _sorted_ok(self, grounds: str, question: str) -> bool:
        """
        順位の問いで、根拠の数値が問いの順になっているかを返す.

        位・番目・順位を含むときだけ見る. 小さい・最小・下位・少ないなら昇順、
        それ以外は降順.

        Args:
            grounds (str): 答えの行より前の本文.
            question (str): HakushoBench の質問.

        Returns:
            bool: 順位の問いでない、または並びが正しければ True.

        Example:
            ok = ChartReadout()._sorted_ok("38.5% 9.1% 11.5%", "4位の国はどこか。")

        """
        if self._rank_q.search(question) is None:
            return True
        ascend = self._ascend_q.search(question) is not None
        for values in self._value_groups(grounds):
            ordered = sorted(values) if ascend else sorted(values, reverse=True)
            if values != ordered:
                return False
        return True

    def _country_other_ok(self, grounds: str, answer: str, question: str) -> bool:
        """
        国名の順位で、「その他」を国として使っていないかを返す.

        相手国、または順位と国が両方あるときだけ見る. 答えが「その他」なら不可.
        根拠に「その他」があっても、除く旨があれば可.

        Args:
            grounds (str): 答えの行より前の本文.
            answer (str): 「答え:」の右側.
            question (str): HakushoBench の質問.

        Returns:
            bool: 対象の問いでない、またはその他を国にしていなければ True.

        Example:
            ok = ChartReadout()._country_other_ok(
                "その他 9.4% バーレーン 9.1%",
                "バーレーン",
                "4位の相手国",
            )

        """
        if self._country_rank_q.search(question) is None:
            return True
        if self.norm(answer).startswith("その他"):
            return False
        if self._other.search(grounds) is None:
            return True
        return self._exclude_other.search(grounds) is not None

    def judge(self, text: str, gold: str, question: str = "") -> tuple[str | None, str]:
        """
        教師の生成を、SFT に残す短い文か、捨てる理由にする.

        残すのは、答えの右側が正解と一致し、根拠が12行以内で、400字以内のときだけ.
        順位の問いは数値が問いの順であること. 国名の順位はその他を国にしないこと.
        思考タグは残さない.

        Args:
            text (str): 思考オンのモデル出力.
            gold (str): HakushoBench の正解.
            question (str): 質問. 空なら順位とその他の検査はしない.

        Returns:
            tuple[str | None, str]: 残す文と `"kept"`. 捨てるときは None と
            `"no_answer"`、`"mismatch"`、`"too_long"`、`"unsorted"`、
            `"other_as_country"` のいずれか.

        Example:
            completion, reason = ChartReadout().judge(
                text="38.5%\\n26.4%\\n11.5%\\n9.1%\\n答え: バーレーン\\n",
                gold="バーレーン",
                question="4位の相手国",
            )

        """
        if self.answer_body(text) is None:
            if self.too_long(text):
                return None, "too_long"
            return None, "no_answer"
        if self.too_long(text):
            return None, "too_long"
        if self.answer_body(text) != self.norm(gold):
            return None, "mismatch"
        grounds = self.grounds(text)
        answer = self.answer_body(text) or ""
        if not self._sorted_ok(grounds, question):
            return None, "unsorted"
        if not self._country_other_ok(grounds, answer, question):
            return None, "other_as_country"
        return "\n".join(self.visible_lines(text)) + "\n", "kept"
