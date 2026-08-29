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


def _format_price(v):
    try:
        return f"{int(v):,} 円"
    except:
        return ""


def _format_market_cap(v):
    try:
        trillion = v / 1e12
        return f"{trillion:,.2f} 兆円"
    except:
        return ""


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
                    "株価_raw": float(price),
                    "時価総額_raw": float(market_cap),
                }
            )
        except Exception:
            continue

    timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    if not rows:
        df_disp = pd.DataFrame(
            columns=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"]
        )
        status = f"{timestamp} 時点 | 取得件数: 0 / 取得成功: 0"
        return df_disp, status

    df_raw = pd.DataFrame(rows)
    df_raw = (
        df_raw.sort_values("時価総額_raw", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    df_raw.insert(0, "順位", range(1, len(df_raw) + 1))

    df_disp = pd.DataFrame()
    df_disp["順位"] = df_raw["順位"]
    df_disp["銘柄コード"] = df_raw["銘柄コード"]
    df_disp["銘柄名"] = df_raw["銘柄名"]
    df_disp["株価（円）"] = df_raw["株価_raw"].apply(_format_price)
    df_disp["時価総額（兆円）"] = df_raw["時価総額_raw"].apply(_format_market_cap)

    status = f"{timestamp} 時点 | 取得件数: {len(df_raw)} / 取得成功: {len(rows)}"
    return df_disp, status


def update():
    return fetch_top10()


def build_interface():
    with gr.Blocks(title="日本株 時価総額 Top10", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 日本株 時価総額 Top10")
        ts_display = gr.Markdown("")
        table = gr.Dataframe(
            headers=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"],
            datatype=["number", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=(10, "fixed"),
        )
        btn = gr.Button("更新", variant="primary", size="lg")
        btn.click(fn=update, inputs=[], outputs=[table, ts_display])
        demo.load(fn=update, outputs=[table, ts_display])
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
