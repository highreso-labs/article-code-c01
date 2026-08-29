import gradio as gr
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

# 候補ユニバース（固定リスト）
UNIVERSE = {
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


def fetch_top_10():
    data_list = []

    for ticker_symbol, name_jp in UNIVERSE.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            market_cap = info.get("marketCap")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")

            if market_cap is not None and current_price is not None:
                data_list.append(
                    {
                        "銘柄コード": ticker_symbol,
                        "銘柄名": name_jp,
                        "株価": current_price,
                        "時価総額": market_cap,
                    }
                )
        except Exception:
            continue

    if not data_list:
        jst = pytz.timezone("Asia/Tokyo")
        now_jst = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S JST")
        return f"**{now_jst}**\n❌ データの取得に失敗しました。", None

    # 生の数値データで降順ソートし、上位10件を抽出
    df = pd.DataFrame(data_list)
    df = df.sort_values(by="時価総額", ascending=False).head(10)

    # 順位列の追加
    df.insert(0, "順位", range(1, len(df) + 1))

    # 表示用フォーマットへの変換
    # 株価: 3桁カンマ + 円
    df["株価"] = df["株価"].apply(lambda x: f"{x:,.0f} 円")
    # 時価総額: 兆円換算 (10^12) + 3桁カンマ + 小数第2位
    df["時価総額"] = df["時価総額"].apply(
        lambda x: f"{x / 1_000_000_000_000:,.2f} 兆円"
    )

    # 列名の変更
    df = df.rename(columns={"株価": "株価（円）", "時価総額": "時価総額（兆円）"})

    # 日本時間の取得
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S JST")
    status_text = f"**{now_jst} 時点**\n✅ 取得件数: {len(df)} 件"

    return status_text, df


def update_app():
    status, df = fetch_top_10()
    return status, df


# カスタムCSSでデザインを調整
custom_css = """
.container { max-width: 900px; margin: auto; }
.status-box { text-align: right; font-family: 'Noto Sans JP', sans-serif; color: #666; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 📈 日本株 時価総額 Top10
        候補ユニバース（31銘柄）の中から、現在の時価総額が高い上位10社を表示します。
        """
    )

    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("### 銘柄ランキング")
        with gr.Column(scale=2):
            update_btn = gr.Button("🔄 更新", variant="primary")

    with gr.Row():
        time_display = gr.Markdown(elem_classes=["status-box"])

    with gr.Row():
        df_display = gr.Dataframe(
            interactive=False,
            wrap=True,
            headers=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"],
        )

    # 起動時にデータを読み込む
    demo.load(update_app, outputs=[time_display, df_display])
    # ボタンクリック時に更新
    update_btn.click(update_app, outputs=[time_display, df_display])

if __name__ == "__main__":
    demo.launch()
