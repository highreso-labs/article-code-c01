import gradio as gr
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta

# 日本時間 (JST) タイムゾーン
JST = timezone(timedelta(hours=9))

# 候補ユニバース (コード: 日本語名)
UNIVERSE = {
    "7203.T": "トヨタ自動車",
    "8306.T": "三菱UFJフィナンシャル・グループ",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "6501.T": "日立製作所",
    "8035.T": "東京エレクトロン",
    "8316.T": "三井住友フィナンシャル・グループ",
    "9984.T": "ソフトバンクグループ",  # 重複チェック: 9984.T は既に存在。リストを確認。
    # リスト再確認:
    # 7203.T トヨタ自動車
    # 8306.T 三菱UFJフィナンシャル・グループ
    # 6758.T ソニーグループ
    # 9984.T ソフトバンクグループ
    # 6501.T 日立製作所
    # 8035.T 東京エレクトロン
    # 8316.T 三井住友フィナンシャル・グループ
    # 9983.T ファーストリテイリング
    # 6857.T アドバンテスト
    # 8411.T みずほフィナンシャル・グループ
    # 4063.T 信越化学工業
    # 6098.T リクルートホールディングス
    # 8058.T 三菱商事
    # 8001.T 伊藤忠商事
    # 8031.T 三井物産
    # 6861.T キーエンス
    # 9432.T 日本電信電話
    # 9433.T KDDI
    # 7011.T 三菱重工業
    # 7974.T 任天堂
    # 4568.T 第一三共
    # 4519.T 中外製薬
    # 6367.T ダイキン工業
    # 8766.T 東京海上ホールディングス
    # 7741.T HOYA
    # 6902.T デンソー
    # 4543.T テルモ
    # 6954.T ファナック
    # 6594.T ニデック
    # 7267.T ホンダ
}

# 上記の重複を除去し、正しい辞書を作成
UNIVERSE = {
    "7203.T": "トヨタ自動車",
    "8306.T": "三菱UFJフィナンシャル・グループ",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "6501.T": "日立製作所",
    "8035.T": "東京エレクトロン",
    "8316.T": "三井住友フィナンシャル・グループ",
    "9983.T": "ファーストリテイリング",
    "6857.T": "アドバンテスト",
    "8411.T": "みずほフィナンシャル・グループ",
    "4063.T": "信越化学工業",
    "6098.T": "リクルートホールディングス",
    "8058.T": "三菱商事",
    "8001.T": "伊藤忠商事",
    "8031.T": "三井物産",
    "6861.T": "キーエンス",
    "9432.T": "日本電信電話",
    "9433.T": "KDDI",
    "7011.T": "三菱重工業",
    "7974.T": "任天堂",
    "4568.T": "第一三共",
    "4519.T": "中外製薬",
    "6367.T": "ダイキン工業",
    "8766.T": "東京海上ホールディングス",
    "7741.T": "HOYA",
    "6902.T": "デンソー",
    "4543.T": "テルモ",
    "6954.T": "ファナック",
    "6594.T": "ニデック",
    "7267.T": "ホンダ",
    "285A.T": "キオクシアホールディングス",
}


def get_top10_market_cap():
    """
    候補ユニバースから時価総額Top10を取得し、Dataframeと取得日時を返す。
    """
    results = []

    # 各銘柄のデータを取得
    for ticker, name in UNIVERSE.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # 必要なデータが存在するか確認
            if "marketCap" in info and "currentPrice" in info:
                market_cap = info["marketCap"]
                price = info["currentPrice"]

                # 時価総額がNoneでないことを確認
                if market_cap is not None and price is not None:
                    results.append(
                        {
                            "銘柄コード": ticker,
                            "銘柄名": name,
                            "株価": price,
                            "時価総額": market_cap,
                        }
                    )
        except Exception as e:
            # 取得失敗した銘柄は除外
            continue

    # 時価総額の降順でソート
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="時価総額", ascending=False)

        # 上位10件だけ取得
        df_top10 = df.head(10)

        # 順位を追加
        df_top10.insert(0, "順位", range(1, len(df_top10) + 1))

        # 列の順序を指定
        df_top10 = df_top10[["順位", "銘柄コード", "銘柄名", "株価", "時価総額"]]

        # 取得日時 (日本時間)
        now_jst = datetime.now(JST)
        timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S") + " JST 時点"

        return df_top10, timestamp_str
    else:
        # データが取得できなかった場合
        empty_df = pd.DataFrame(
            columns=["順位", "銘柄コード", "銘柄名", "株価", "時価総額"]
        )
        now_jst = datetime.now(JST)
        timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S") + " JST 時点"
        return empty_df, timestamp_str


def update_data():
    """
    Gradioの更新ボタン用関数。
    """
    df, timestamp = get_top10_market_cap()
    return df, timestamp


# Gradio UI 構築
with gr.Blocks(title="日本株の時価総額Top10") as app:
    gr.Markdown("# 日本株の時価総額 Top 10")

    # 取得日時表示
    timestamp_md = gr.Markdown("")

    # データフレーム表示
    df_display = gr.Dataframe(
        headers=["順位", "銘柄コード", "銘柄名", "株価", "時価総額"], interactive=False
    )

    # 更新ボタン
    update_btn = gr.Button("更新")

    # イベント設定
    update_btn.click(fn=update_data, outputs=[df_display, timestamp_md])

    # 起動時に1回取得
    app.load(fn=update_data, outputs=[df_display, timestamp_md])

if __name__ == "__main__":
    app.launch()
