import yfinance as yf
import gradio as gr
import pandas as pd
from datetime import datetime
import pytz

UNIVERSE = [
    ("7203.T", "トヨタ自動車"),
    ("8306.T", "三菱UFJフィナンシャル・グループ"),
    ("6758.T", "ソニーグループ"),
    ("9984.T", "ソフトバンクグループ"),
    ("6501.T", "日立製作所"),
    ("8035.T", "東京エレクトロン"),
    ("8316.T", "三井住友フィナンシャルグループ"),
    ("9983.T", "ファーストリテイリング"),
    ("6857.T", "アドバンテスト"),
    ("8411.T", "みずほフィナンシャルグループ"),
    ("4063.T", "信越化学工業"),
    ("6098.T", "リクルートホールディングス"),
    ("8058.T", "三菱商事"),
    ("8001.T", "伊藤忠商事"),
    ("8031.T", "三井物産"),
    ("6861.T", "キーエンス"),
    ("9432.T", "日本電信電話"),
    ("9433.T", "KDDI"),
    ("7011.T", "三菱重工業"),
    ("7974.T", "任天堂"),
    ("4568.T", "第一三共"),
    ("4519.T", "中外製薬"),
    ("6367.T", "ダイキン工業"),
    ("8766.T", "東京海上ホールディングス"),
    ("7741.T", "HOYA"),
    ("6902.T", "デンソー"),
    ("4543.T", "テルモ"),
    ("6954.T", "ファナック"),
    ("6594.T", "ニデック"),
    ("7267.T", "ホンダ"),
    ("285A.T", "キオクシアホールディングス"),
]

JST = pytz.timezone("Asia/Tokyo")


def fetch_top10():
    rows = []
    for code, name_jp in UNIVERSE:
        try:
            ticker = yf.Ticker(code)
            info = ticker.info
            market_cap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if market_cap is None or price is None:
                continue
            rows.append(
                {
                    "銘柄コード": code,
                    "銘柄名": name_jp,
                    "株価": float(price),
                    "時価総額": float(market_cap),
                }
            )
        except Exception:
            continue

    if not rows:
        df = pd.DataFrame(columns=["順位", "銘柄コード", "銘柄名", "株価", "時価総額"])
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
        return df, f"{timestamp} 時点"

    df = pd.DataFrame(rows)
    df = df.sort_values("時価総額", ascending=False).head(10).reset_index(drop=True)
    df.insert(0, "順位", range(1, len(df) + 1))
    df = df[["順位", "銘柄コード", "銘柄名", "株価", "時価総額"]]
    timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    return df, f"{timestamp} 時点"


def update():
    df, ts = fetch_top10()
    return df, ts


def build_interface():
    df_init, ts_init = fetch_top10()
    with gr.Blocks(title="日本株 時価総額 Top10") as demo:
        ts_display = gr.Markdown(f"### {ts_init}")
        table = gr.Dataframe(
            value=df_init,
            headers=["順位", "銘柄コード", "銘柄名", "株価", "時価総額"],
            datatype=["number", "str", "str", "number", "number"],
            wrap=True,
        )
        btn = gr.Button("更新")
        btn.click(fn=update, inputs=[], outputs=[table, ts_display])
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
