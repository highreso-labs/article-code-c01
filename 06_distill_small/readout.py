"""図表の読み取り型. 思考チャネルで段階的に読み取り、最後の行を「答え:」で始める."""

from __future__ import annotations

import re
import unicodedata


class ChartReadout:
    """
    日本の白書図表 VQA の思考・根拠・回答の型を、プロンプトと採点で同じ規則にする.

    思考モード（`<|channel>thought...<channel|>` または `<think>...</think>`）で
    図表のタイトル・単位・注記・全対象項目の数値を日本語で段階的に確認させ、
    本文には簡潔な根拠と「答え:」の行を書かせる.

    Attributes:
        MARK (str): 答えの行の先頭. 「答え:」.
        MAX_GROUNDS (int): 答えの行より前に許す空でない行数. 40.
        MAX_CHARS (int): 本文の空白を除いた文字数の上限. 2000.
        MAX_THOUGHT_CHARS (int): 思考の文字数の上限. 6000.

    Methods:
        prompt: 質問を、日本語思考とこの回答型を求めるユーザー文にする.
        norm: 比較用に表記（太字記号・全角・波ダッシュ・空白・句点）を揃える.
        extract_thought: 思考チャネルまたは思考タグの中身を返す.
        visible: 思考より後の本文を返す.
        visible_lines: 思考を除いた本文を、空でない行のリストにして返す.
        find_answer_index: 「答え:」で始まる最後の行の位置を返す.
        answer_body: 「答え:」の右側を正規化して返す.
        grounds: 答えの行より前の根拠を返す.
        format_assistant: 思考と本文を Gemma 4 の `<|channel>thought` 形式に結合する.
        too_long: 思考や本文が上限を超えているか返す.
        answer_closed: 思考を閉じたあとに「答え:」の行が改行まで出ているか返す.
        judge: 生成文を、SFT に残す（思考, 本文）か捨てる理由にする.

    Example:
        readout = ChartReadout()
        user_prompt = readout.prompt(question="4位の国はどこか。")
        completion, reason = readout.judge(
            text="<think>1位は中国...</think>\\nバーレーン\\n答え: バーレーン\\n",
            gold="バーレーン",
            question="4位の国はどこか。",
        )

    """

    MARK = "答え:"
    MAX_GROUNDS = 40
    MAX_CHARS = 2000
    MAX_THOUGHT_CHARS = 6000

    _rank_q = re.compile(r"位|番目|順位")
    _country_rank_q = re.compile(
        r"相手国|(?:位|番目|順位).{0,8}国|国.{0,8}(?:位|番目|順位)"
    )
    _other = re.compile(r"その他")
    _exclude_other = re.compile(
        r"その他.{0,16}(?:除|含めない|含まない|みなさない|外[すし])"
        r"|(?:除|含めない|含まない|みなさない|外[すし]).{0,16}その他"
    )
    _jp_char = re.compile(r"[ぁ-んァ-ヶ一-龠々]")
    _eng_thought_prefix = re.compile(r"^\s*(?:Thinking Process|Here's a thinking process)", re.IGNORECASE)

    def prompt(self, question: str) -> str:
        """
        図表の質問を、日本語での段階的思考と根拠・「答え:」を求めるユーザー文にする.

        Args:
            question (str): HakushoBench の質問.

        Returns:
            str: 思考手順の指示と質問を連結した文.

        Example:
            user_prompt = ChartReadout().prompt(question="4位の国はどこか。")

        """
        rank_rule = (
            "順位や最大・最小を問う場合は、まず図表の並び順のまま各項目の数値を読み取って書き出し、そのあとで大小を比較して順位を特定してください。"
            if self._rank_q.search(question)
            else ""
        )
        country_rule = (
            "相手国や国名の順位では、「その他」は国とみなさないでください。"
            if self._country_rank_q.search(question)
            else ""
        )
        return (
            "この図表を読み、思考も日本語で行って回答してください。\n"
            "思考では、(1)図表のタイトル・サブグラフ番号・凡例の色や線種・軸の単位・注記・括弧書き(n=...等)の確認、"
            "(2)関係するすべての項目・年次の数値の書き出しと計算、"
            "(3)質問が求める単位・表記（年/年度、%/ポイント、億円以上など）と端数処理の確認（指示がなければ勝手に四捨五入しない）を順に行ってください。"
            + rank_rule
            + country_rule
            + "\n本文には答えに必要な簡潔な根拠（項目・数値・単位・比較）を書き、"
            "最後の行を「答え:」で始め、その右側に問いの短い答えだけを書いてください（太字記号 ** は使わないでください）。\n"
            + question
        )

    def norm(self, text: str) -> str:
        """
        太字記号・全角・空白・波ダッシュ・ハイフン・囲み括弧・末尾の句点を揃えて、比較用の文字列を返す.

        Args:
            text (str): 正解、またはモデルの出力.

        Returns:
            str: NFKC 正規化し、`*` と空白と末尾の句点を除いた文字列.
            `〜` `～` `-` は同じ. 答え全体を囲む `()` は除く.

        Example:
            key = ChartReadout().norm("**(35〜39歳)**")

        """
        folded = unicodedata.normalize("NFKC", text)
        folded = folded.replace("*", "")
        for mark in ("〜", "～", "~", "―", "–", "—", "−"):
            folded = folded.replace(mark, "-")
        folded = re.sub(r"\s+", "", folded)
        while True:
            stripped = folded.strip("。．.、,")
            if len(stripped) >= 2 and stripped[0] == "(" and stripped[-1] == ")":
                folded = stripped[1:-1]
                continue
            return stripped

    def extract_thought(self, text: str) -> str:
        """
        モデル出力から思考を取り出す.

        Gemma 4 は `<|channel>thought`、Qwen は `<think>` を使う.
        Qwen の vLLM は開始タグをプロンプト側に置くため、生成文には
        `</think>` より前の本文だけが思考として残る.

        Args:
            text (str): モデル出力.

        Returns:
            str: 思考の本文. 思考ブロックが無ければ空文字.

        Example:
            thought = ChartReadout().extract_thought("項目を読む\\n</think>\\n答え: 10%")

        """
        thoughts = []
        if "</think>" in text and "<think>" not in text:
            thoughts.append(text.split("</think>", 1)[0].strip())
        if "<|channel>thought" in text:
            after_open = text.split("<|channel>thought", 1)[1]
            if "<channel|>" in after_open:
                channel_body = after_open.split("<channel|>", 1)[0]
            else:
                channel_body = after_open
            thoughts.append(channel_body.strip())
        if "<think>" in text:
            after_think = text.split("<think>", 1)[1]
            if "</think>" in after_think:
                think_body = after_think.split("</think>", 1)[0]
            else:
                think_body = after_think
            if think_body.strip() and think_body.strip() not in thoughts:
                thoughts.append(think_body.strip())
        combined = "\n".join(part for part in thoughts if part)
        for token in (
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<turn|>",
            "<|turn>",
            "<|channel>",
            "<channel|>",
            "<bos>",
            "<eos>",
        ):
            combined = combined.replace(token, "")
        return combined.strip()

    def visible(self, text: str) -> str:
        """
        思考チャネル・思考タグより後の本文を返す. 思考が閉じるまでは空にする.

        Args:
            text (str): モデル出力. 思考チャネルや思考タグを含み得る.

        Returns:
            str: `<channel|>` または `</think>` より後. 思考の途中なら空文字.

        Example:
            body = ChartReadout().visible("<|channel>thought\\n読む<channel|>答え: 3.4倍")

        """
        body = text
        if "<|channel>thought" in body:
            if "<channel|>" not in body:
                return ""
            body = body.split("<channel|>", 1)[1]
        if "<think>" in body:
            if "</think>" not in body:
                return ""
            body = body.split("</think>", 1)[1]
        elif "</think>" in body:
            body = body.split("</think>", 1)[1]
        for token in (
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<turn|>",
            "<|turn>",
            "<|channel>",
            "<channel|>",
            "<bos>",
            "<eos>",
        ):
            body = body.replace(token, "")
        return body.strip()

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

    def find_answer_index(self, lines: list[str]) -> int | None:
        """
        「答え:」で始まる最後の行の位置を返す.

        Args:
            lines (list[str]): `visible_lines` の結果.

        Returns:
            int | None: 行の位置. 無ければ None.

        Example:
            index = ChartReadout().find_answer_index(["項目", "**答え:** 3.4倍"])

        """
        for index in range(len(lines) - 1, -1, -1):
            if self.norm(lines[index]).startswith(self.MARK):
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
            body = ChartReadout().answer_body("項目\\n**答え:** (3.4倍)\\n")

        """
        lines = self.visible_lines(text)
        index = self.find_answer_index(lines)
        if index is None:
            return None
        folded = self.norm(lines[index])
        return self.norm(folded[len(self.MARK) :])

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
        index = self.find_answer_index(lines)
        if index is None:
            return "\n".join(lines)
        return "\n".join(lines[:index])

    def format_assistant(self, thinking: str, completion: str) -> str:
        """
        思考と本文を Gemma 4 の思考チャネル形式に結合して返す.

        Args:
            thinking (str): 思考プロセス.
            completion (str): 根拠と「答え:」の行.

        Returns:
            str: `<|channel>thought\\n...\\n<channel|>...` 形式の文字列.

        Example:
            assistant_text = ChartReadout().format_assistant("数値を比較する。", "根拠: ...\\n答え: 地域C\\n")

        """
        clean_thinking = thinking.strip()
        clean_completion = completion.strip() + "\n"
        if not clean_thinking:
            return clean_completion
        return f"<|channel>thought\n{clean_thinking}\n<channel|>{clean_completion}"

    def too_long(self, text: str) -> bool:
        """
        思考や根拠が上限を超えるか、答えの行の後に余計な文があるかを返す.

        Args:
            text (str): モデル出力.

        Returns:
            bool: 上限を超えていれば True.

        Example:
            long = ChartReadout().too_long("a\\n" * 25 + "答え: 5回\\n")

        """
        thought = self.extract_thought(text)
        if len(thought) > self.MAX_THOUGHT_CHARS:
            return True
        thought_lines = [line.strip() for line in thought.splitlines() if line.strip()]
        if len(thought_lines) >= 10 and len(set(thought_lines)) <= len(thought_lines) // 3:
            return True
        lines = self.visible_lines(text)
        index = self.find_answer_index(lines)
        grounds_lines = lines if index is None else lines[:index]
        if len(grounds_lines) > self.MAX_GROUNDS:
            return True
        if index is not None and index != len(lines) - 1:
            return True
        return len(self.norm("\n".join(lines))) > self.MAX_CHARS

    def answer_closed(self, text: str) -> bool:
        """
        思考を閉じたあとの生成文に、「答え:」の行が改行まで出ているかを返す.

        Args:
            text (str): そこまでの生成をデコードした文.

        Returns:
            bool: 思考外の答えの印の後に改行があれば True.

        Example:
            closed = ChartReadout().answer_closed("<|channel>thought\\n読む<channel|>項目\\n答え: 3.4倍\\n")

        """
        visible_text = self.visible(text)
        if not visible_text:
            return False
        for mark in (self.MARK, "答え："):
            at = visible_text.rfind(mark)
            if at >= 0 and "\n" in text[text.rfind(mark) :]:
                return True
        return False

    def judge(self, text: str, gold: str, question: str = "") -> tuple[str | None, str]:
        """
        生成文を採点し、SFT に残す本文か、捨てる理由を返す.

        残すのは、日本語の思考があり、答えの右側が正解と一致し、長さ上限以内のときだけ.
        国名の順位では「その他」を国にしないこと.
        太字の `**答え:**` は正規化した `答え: ...` に整形して残す.

        Args:
            text (str): 思考オンのモデル出力.
            gold (str): HakushoBench の正解.
            question (str): 質問.

        Returns:
            tuple[str | None, str]: 残す本文と `"kept"`. 捨てるときは None と
            `"no_thought"`、`"english_thought"`、`"no_answer"`、`"mismatch"`、
            `"too_long"`、`"other_as_country"` のいずれか.

        Example:
            completion, reason = ChartReadout().judge(
                text="<think>1位は中国、4位はバーレーン</think>\\n根拠: ...\\n答え: バーレーン\\n",
                gold="バーレーン",
                question="4位の相手国",
            )

        """
        if self.too_long(text):
            return None, "too_long"
        thought = self.extract_thought(text)
        if not thought:
            thought = self.grounds(text).strip()
        if not thought:
            return None, "no_thought"
        if self._eng_thought_prefix.search(thought) or not self._jp_char.search(thought):
            return None, "english_thought"
        body = self.answer_body(text)
        if body is None:
            return None, "no_answer"
        if body != self.norm(gold):
            return None, "mismatch"
        grounds_text = self.grounds(text).replace("**", "").strip()
        if self._country_rank_q.search(question) is not None:
            if self.norm(body).startswith("その他"):
                return None, "other_as_country"
            if (
                self._other.search(grounds_text) is not None
                and self._exclude_other.search(grounds_text) is None
            ):
                return None, "other_as_country"
        lines = self.visible_lines(text)
        answer_index = self.find_answer_index(lines)
        raw_answer_line = lines[answer_index].replace("*", "").strip()
        raw_right = re.sub(r"^答え\s*[:：]\s*", "", raw_answer_line).strip()
        clean_lines = [line.replace("**", "").strip() for line in lines[:answer_index]]
        clean_lines.append(f"答え: {raw_right}")
        return "\n".join(clean_lines) + "\n", "kept"
