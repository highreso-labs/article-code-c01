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

    # 時価総額で降順ソートし、上位10件を抽出
    df = pd.DataFrame(data_list)
    if df.empty:
        return "データが取得できませんでした", None

    df = df.sort_values(by="時価総額", ascending=False).head(10)

    # 順位列の追加
    df.insert(0, "順位", range(1, len(df) + 1))

    # 日本時間の取得
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S JST")

    return now_jst, df


def update_app():
    timestamp, df = fetch_top_10()
    return timestamp, df


with gr.Blocks() as demo:
    gr.Markdown("# 日本株 時価総額 Top10")

    with gr.Row():
        time_display = gr.Markdown()

    with gr.Row():
        df_display = gr.Dataframe()

    update_btn = gr.Button("更新")

    # 初期ロード
    demo.load(update_app, outputs=[time_display, df_display])
    # ボタンクリック時の更新
    update_btn.click(update_app, outputs=[time_display, df_display])

if __name__ == "__main__":
    demo.launch()
