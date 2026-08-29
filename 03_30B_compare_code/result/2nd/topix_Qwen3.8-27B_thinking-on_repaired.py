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

    # 生の数値データで時価総額の降順ソート（フォーマット前）
    results.sort(key=lambda x: x["market_cap"], reverse=True)
    top10 = results[:10]

    rows = []
    for i, item in enumerate(top10, 1):
        rows.append(
            {
                "順位": i,
                "銘柄コード": item["code"],
                "銘柄名": item["name"],
                "株価（円）": f"{item['price']:,.0f}",
                "時価総額（兆円）": f"{item['market_cap'] / 1e12:,.2f}",
            }
        )

    df = pd.DataFrame(rows)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    count = len(df)
    return timestamp, count, df


def on_refresh():
    ts, count, df = fetch_top10()
    status = (
        f"📊 取得日時: **{ts} JST**  |  "
        f"取得件数: **{count} 件** / 候補 {len(UNIVERSE)} 銘柄  |  "
        f"ステータス: ✅ 正常"
    )
    return status, df


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#f0f4ff",
        c100="#dbe4ff",
        c200="#bcccff",
        c300="#93aaff",
        c400="#6b8cff",
        c500="#4a6cf7",
        c600="#3b52d9",
        c700="#3243b3",
        c800="#2d3a94",
        c900="#2b3678",
        c950="#1e224d",
    ),
    secondary_hue="slate",
    neutral_hue="slate",
    font_google=[{"name": "Noto Sans JP", "subset": "japanese"}],
)

custom_css = """
<style>
#status-box {
    padding: 12px 16px;
    border-radius: 8px;
    background: linear-gradient(135deg, #f0f4ff 0%, #e8ecf7 100%);
    border: 1px solid #bcccff;
    font-size: 14px;
}
#main-title {
    text-align: center;
    font-size: 1.8em;
    font-weight: 700;
    color: #2d3a94;
    margin-bottom: 4px;
}
#subtitle {
    text-align: center;
    font-size: 0.9em;
    color: #64748b;
    margin-bottom: 16px;
}
.gradio-container {
    max-width: 900px;
    margin: auto;
}
</style>
"""

with gr.Blocks(title="日本株 時価総額 Top10", theme=theme, css=custom_css) as demo:
    gr.Markdown("# 📈 日本株 時価総額 Top10", elem_id="main-title")
    gr.Markdown(
        "候補ユニバース 31 銘柄から時価総額上位 10 銘柄を表示", elem_id="subtitle"
    )

    status_md = gr.Markdown("⏳ 読み込み中...", elem_id="status-box")

    df_display = gr.Dataframe(
        value=None,
        headers=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"],
        interactive=False,
    )

    refresh_btn = gr.Button("🔄 更新", variant="primary", size="lg")

    refresh_btn.click(fn=on_refresh, outputs=[status_md, df_display])
    demo.load(fn=on_refresh, outputs=[status_md, df_display])

demo.queue().launch()
