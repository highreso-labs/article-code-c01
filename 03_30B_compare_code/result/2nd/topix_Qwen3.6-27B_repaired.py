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

    # yfinanceで一括取得を試みる
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
            # 取得失敗した銘柄は除外
            continue

    if not results:
        return pd.DataFrame(), "データ取得に失敗しました。", 0

    # DataFrameに変換
    df = pd.DataFrame(results)

    # 【最重要】生の数値データで時価総額の降順ソートを行う
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
            "price": "株価（円）",
            "market_cap": "時価総額（兆円）",
        }
    )

    # 数値のフォーマット（可読性向上）
    # 株価は3桁区切り整数 + 円
    df_display["株価（円）"] = df_display["株価（円）"].apply(lambda x: f"{x:,.0f} 円")

    # 時価総額は兆円単位（1兆 = 10^12）に換算し、小数第2位まで表示 + 3桁区切り
    df_display["時価総額（兆円）"] = df_display["時価総額（兆円）"].apply(
        lambda x: f"{x / 1e12:,.2f} 兆円"
    )

    # 現在の日本時間を取得
    now_jst = datetime.now(JST)
    timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S JST 時点")

    return df_display, timestamp_str, len(df_top10)


def update_data():
    """
    Gradioの更新ボタン用コールバック
    """
    df, timestamp, count = get_top10_market_cap()

    if df.empty:
        status_msg = f"{timestamp} | 取得件数: 0 | ステータス: 取得失敗"
    else:
        status_msg = f"{timestamp} | 取得件数: {count} | ステータス: 正常"

    return status_msg, df


# Gradio UIの構築
with gr.Blocks(
    title="日本株 時価総額 Top10",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
    .gradio-container {
        max-width: 900px !important;
    }
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
    }
    """,
) as app:
    gr.Markdown(
        """
        # 📈 日本株 時価総額 Top10
        候補ユニバース（31銘柄）から時価総額上位10銘柄を表示します。
        """
    )

    # 日時・ステータス表示
    status_text = gr.Textbox(label="ステータス", interactive=False, lines=1)

    # データフレーム表示
    df_output = gr.Dataframe(
        label="時価総額 Top10",
        headers=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"],
        interactive=False,
        wrap=True,
    )

    # 更新ボタン
    update_btn = gr.Button("🔄 データ更新", variant="primary", size="lg")

    # イベント設定
    update_btn.click(fn=update_data, outputs=[status_text, df_output])

    # 起動時に一度だけ取得（重い処理をUI起動後に遅延実行）
    app.load(fn=update_data, outputs=[status_text, df_output])

if __name__ == "__main__":
    app.launch()
