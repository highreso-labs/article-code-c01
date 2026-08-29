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
        except Exception:
            # 取得失敗した銘柄は除外
            continue

    # 時価総額の降順でソート
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="時価総額", ascending=False)

        # 上位10件だけ取得
        df_top10 = df.head(10).copy()

        # 順位を追加
        df_top10.insert(0, "順位", range(1, len(df_top10) + 1))

        # 列の順序を指定
        df_top10 = df_top10[["順位", "銘柄コード", "銘柄名", "株価", "時価総額"]]

        # 取得日時 (日本時間)
        now_jst = datetime.now(JST)
        timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S") + " JST 時点"

        # フォーマット処理
        # 株価: 3桁区切り + 円
        df_top10["株価"] = df_top10["株価"].apply(lambda x: f"{x:,.0f} 円")

        # 時価総額: 兆円換算 (1兆 = 1e12), 小数第2位, 3桁区切り
        df_top10["時価総額"] = df_top10["時価総額"].apply(
            lambda x: f"{x / 1e12:,.2f} 兆円"
        )

        # 列名の変更
        df_top10.columns = [
            "順位",
            "銘柄コード",
            "銘柄名",
            "株価（円）",
            "時価総額（兆円）",
        ]

        return df_top10, timestamp_str
    else:
        # データが取得できなかった場合
        empty_df = pd.DataFrame(
            columns=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"]
        )
        now_jst = datetime.now(JST)
        timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S") + " JST 時点"
        return empty_df, timestamp_str


def update_data():
    """
    Gradioの更新ボタン用関数。
    """
    df, timestamp = get_top10_market_cap()

    # 件数とステータスの表示文字列を生成
    if not df.empty:
        count = len(df)
        status_msg = f"✅ 正常に {count} 件のデータを取得しました。"
    else:
        count = 0
        status_msg = "❌ データの取得に失敗しました。"

    header_md = f"""
    <div style="background-color: #f0f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; margin-bottom: 10px;">
        <span style="font-weight: bold; color: #2c3e50;">📅 取得日時:</span> {timestamp}<br>
        <span style="font-weight: bold; color: #2c3e50;">📊 取得件数:</span> {count} 件<br>
        <span style="font-weight: bold; color: #2c3e50;">📡 ステータス:</span> {status_msg}
    </div>
    """
    return df, header_md


# Gradio UI 構築
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#f0f9ff",
        c100="#e0f2fe",
        c200="#bae6fd",
        c300="#7dd3fc",
        c400="#38bdf8",
        c500="#0ea5e9",
        c600="#0284c7",
        c700="#0369a1",
        c800="#075985",
        c900="#0c4a6e",
        c950="#082f49",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f8fafc",
        c100="#f1f5f9",
        c200="#e2e8f0",
        c300="#cbd5e1",
        c400="#94a3b8",
        c500="#64748b",
        c600="#475569",
        c700="#334155",
        c800="#1e293b",
        c900="#0f172a",
        c950="#020617",
    ),
    font_mono="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
    font_google=[
        "Noto Sans JP",
    ],
)

with gr.Blocks(title="日本株の時価総額Top10", theme=theme) as app:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #1e293b; font-weight: 700;">📈 日本株の時価総額 Top 10</h1>
            <p style="color: #64748b; font-size: 14px;">主要31銘柄からリアルタイムで時価総額上位10社を表示します</p>
        </div>
        """
    )

    # 取得日時・ステータス表示
    timestamp_md = gr.Markdown("")

    # データフレーム表示
    df_display = gr.Dataframe(
        headers=["順位", "銘柄コード", "銘柄名", "株価（円）", "時価総額（兆円）"],
        interactive=False,
        wrap=True,
        row_count=(10, "fixed"),
        col_count=(5, "fixed"),
        type="pandas",
        label="時価総額ランキング",
    )

    # 更新ボタン
    update_btn = gr.Button("🔄 データを更新", variant="primary", size="lg")

    # イベント設定
    update_btn.click(fn=update_data, outputs=[df_display, timestamp_md])

    # 起動時に1回取得
    app.load(fn=update_data, outputs=[df_display, timestamp_md])

if __name__ == "__main__":
    app.launch()
