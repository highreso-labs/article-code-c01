import gradio as gr
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta

# 日本時間のオフセット (UTC+9)
JST = timezone(timedelta(hours=9))

# 候補ユニバース（指定されたリストをそのまま使用）
UNIVERSE = [
    "7203.T",
    "8306.T",
    "6758.T",
    "9984.T",
    "6501.T",
    "8035.T",
    "8316.T",
    "9983.T",
    "6857.T",
    "8411.T",
    "4063.T",
    "6098.T",
    "8058.T",
    "8001.T",
    "8031.T",
    "6861.T",
    "9432.T",
    "9433.T",
    "7011.T",
    "7974.T",
    "4568.T",
    "4519.T",
    "6367.T",
    "8766.T",
    "7741.T",
    "6902.T",
    "4543.T",
    "6954.T",
    "6594.T",
    "7267.T",
    "285A.T",
]

# 銘柄コードと日本語名の対応辞書
TICKER_NAMES = {
    "7203.T": "トヨタ自動車",
    "8306.T": "三菱UFJフィナンシャル・グループ",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "6501.T": "日立製作所",
    "8035.T": "東京エレクトロン",
    "8316.T": "三井住友フィナンシャルグループ",
    "9983.T": "ファーストリテイリング",
    "6857.T": "アドバンテスト",
    "8411.T": "みずほフィナンシャルグループ",
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
    候補ユニバースから時価総額を取得し、Top10を返す関数
    """
    results = []

    # yfinanceで一括取得を試みる（効率的）
    # ただし、個別に取得した方がエラーハンドリングが確実な場合もあるため、
    # ここでは一括取得後、欠損値をフィルタリングするアプローチを取る
    tickers = UNIVERSE
    data = yf.Tickers(tickers)

    for ticker in tickers:
        try:
            info = data.tickers[ticker].info
            market_cap = info.get("marketCap")
            current_price = info.get("currentPrice")

            # データが存在し、数値として有効かチェック
            if market_cap is not None and current_price is not None:
                # 時価総額が0以下や異常値の場合は除外
                if market_cap > 0:
                    results.append(
                        {
                            "ticker": ticker,
                            "name": TICKER_NAMES.get(ticker, "Unknown"),
                            "price": current_price,
                            "market_cap": market_cap,
                        }
                    )
        except Exception as e:
            # 取得失敗した銘柄は除外（要件6）
            continue

    if not results:
        return pd.DataFrame(), "データ取得に失敗しました。"

    # DataFrameに変換
    df = pd.DataFrame(results)

    # 時価総額で降順ソート
    df = df.sort_values(by="market_cap", ascending=False)

    # Top10を取得
    df_top10 = df.head(10)

    # 順位列を追加
    df_top10.insert(0, "Rank", range(1, len(df_top10) + 1))

    # 表示用の列名を変更（日本語）
    df_display = df_top10.rename(
        columns={
            "ticker": "銘柄コード",
            "name": "銘柄名",
            "price": "株価 (JPY)",
            "market_cap": "時価総額 (JPY)",
        }
    )

    # 数値のフォーマット（可読性向上）
    # 株価は整数、時価総額は桁区切り
    df_display["株価 (JPY)"] = df_display["株価 (JPY)"].apply(lambda x: f"{x:,.0f}")
    df_display["時価総額 (JPY)"] = df_display["時価総額 (JPY)"].apply(
        lambda x: f"{x:,.0f}"
    )

    # 現在の日本時間を取得
    now_jst = datetime.now(JST)
    timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S JST 時点")

    return df_display, timestamp_str


def update_data():
    """
    Gradioの更新ボタン用コールバック
    """
    df, timestamp = get_top10_market_cap()
    return timestamp, df


# Gradio UIの構築
with gr.Blocks(title="日本株 時価総額 Top10") as app:
    gr.Markdown("# 日本株 時価総額 Top10")
    gr.Markdown("候補ユニバース（30銘柄）から時価総額上位10銘柄を表示します。")

    # 日時表示
    timestamp_text = gr.Textbox(label="データ取得日時", interactive=False)

    # データフレーム表示
    df_output = gr.Dataframe(
        label="時価総額 Top10",
        headers=["順位", "銘柄コード", "銘柄名", "株価 (JPY)", "時価総額 (JPY)"],
    )

    # 更新ボタン
    update_btn = gr.Button("データ更新")

    # イベント設定
    update_btn.click(fn=update_data, outputs=[timestamp_text, df_output])

    # 起動時に一度だけ取得
    app.load(fn=update_data, outputs=[timestamp_text, df_output])

if __name__ == "__main__":
    app.launch()
