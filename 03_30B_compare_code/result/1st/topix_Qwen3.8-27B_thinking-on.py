import yfinance as yf
import gradio as gr
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

UNIVERSE = [
    ("7203.T", "トヨタ自動車"),
    ("8306.T", "三菱UFJフィナンシャル・グループ"),
    ("6758.T", "ソニーグループ"),
    ("9984.T", "ソフトバンクグループ"),
    ("6501.T", "日立製作所"),
    ("8035.T", "東京エレクトロン"),
    ("8316.T", "三井住友フィナンシャル・グループ"),
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


def fetch_top10():
    jst = ZoneInfo("Asia/Tokyo")
    now = datetime.now(jst)

    results = []
    for code, name in UNIVERSE:
        try:
            ticker = yf.Ticker(code)
            info = ticker.info
            market_cap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if market_cap is None or price is None:
                continue
            results.append(
                {
                    "code": code,
                    "name": name,
                    "price": price,
                    "market_cap": market_cap,
                }
            )
        except Exception:
            continue

    results.sort(key=lambda x: x["market_cap"], reverse=True)
    top10 = results[:10]

    rows = []
    for i, item in enumerate(top10, 1):
        rows.append(
            {
                "順位": i,
                "銘柄コード": item["code"],
                "銘柄名": item["name"],
                "株価": f"{item['price']:,.0f}",
                "時価総額": f"{item['market_cap'] / 1e12:,.2f} 兆円",
            }
        )

    df = pd.DataFrame(rows)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + " JST 時点"
    return timestamp, df


def on_refresh():
    ts, df = fetch_top10()
    return f"**{ts}**", df


initial_ts, initial_df = fetch_top10()

with gr.Blocks(title="日本株 時価総額 Top10") as demo:
    gr.Markdown("# 日本株 時価総額 Top10")
    ts_md = gr.Markdown(f"**{initial_ts}**")
    df_display = gr.Dataframe(
        value=initial_df,
        headers=["順位", "銘柄コード", "銘柄名", "株価", "時価総額"],
        interactive=False,
    )
    refresh_btn = gr.Button("更新")
    refresh_btn.click(fn=on_refresh, outputs=[ts_md, df_display])

demo.queue().launch()
