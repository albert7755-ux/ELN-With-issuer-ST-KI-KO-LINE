import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="結構型商品戰情室 (V10.7 - No Profile)", layout="wide")

# ==========================================
# 🔐 密碼保護機制
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        if st.session_state["password"] == "5428":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("請輸入系統密碼 (Access Code)", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("請輸入系統密碼 (Access Code)", type="password", on_change=password_entered, key="password")
        st.error("❌ 密碼錯誤 (Incorrect Password)")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# 🔓 主程式開始
# ==========================================

st.title("📊 FCN - 關鍵點位與長週期風險回測")
st.markdown("回測區間：**2009/01/01 至今**。")
st.divider()

# --- 2. 側邊欄：參數設定 ---
st.sidebar.header("1️⃣ 輸入標的與發行機構")
default_tickers = "TSLA, NVDA, GOOG"
tickers_input = st.sidebar.text_area("股票代碼 (逗號分隔)", value=default_tickers, height=80)

# 保留您的 Issuer 選擇
issuer_list = ["All", "Barclays", "Citi", "BNP", "SG", "UBS", "GS", "MS", "JPM", "BCS", "NOMURA"]
issuer_input = st.sidebar.selectbox("選擇發行機構 (Issuer)", issuer_list, index=0)

st.sidebar.divider()
st.sidebar.header("2️⃣ 結構條件 (%)")
st.sidebar.info("以該期「進場價」為 100% 基準：")

ko_pct = st.sidebar.number_input("KO (敲出價 %)", value=100.0, step=0.5, format="%.1f")
strike_pct = st.sidebar.number_input("Strike (轉換/執行價 %)", value=80.0, step=1.0, format="%.1f")
ki_pct = st.sidebar.number_input("KI (下檔保護價 %)", value=65.0, step=1.0, format="%.1f")

st.sidebar.divider()
st.sidebar.header("3️⃣ 投資與配息設定")
principal = st.sidebar.number_input("投資本金 (例如 USD)", value=100000, step=10000, help="輸入客戶預計投資的金額")
coupon_pa = st.sidebar.number_input("年化配息率 (Coupon %)", value=8.0, step=0.5, format="%.1f")

st.sidebar.divider()
st.sidebar.header("4️⃣ 回測參數設定")
period_months = st.sidebar.number_input("產品/觀察天期 (月)", min_value=1, max_value=60, value=6, step=1)

run_btn = st.sidebar.button("🚀 開始分析", type="primary")

# --- 3. 核心函數 ---

def get_stock_data_from_2009(ticker):
    try:
        start_date = "2009-01-01"
        df = yf.download(ticker, start=start_date, progress=False)
        
        if df.empty: 
            return None, f"找不到 {ticker} 或該期間無資料"
        
        # 💡 核心修復：處理新版 yfinance 回傳的 MultiIndex 欄位結構
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 💡 核心修復：將日期索引重設為 DataFrame 欄位
        df = df.reset_index()
        
        # 💡 核心修復：若新版 yfinance 將日期欄位命名為 'index' 或 'Datetime'，統一改回 'Date'
        if 'Date' not in df.columns:
            df = df.rename(columns={'index': 'Date', 'Datetime': 'Date'})
            
        # 💡 核心修復：移除可能重複的欄位名
        df = df.loc[:, ~df.columns.duplicated()]
        
        if 'Close' not in df.columns: 
            return None, "無收盤價資料"

        df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        # 均線計算
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA240'] = df['Close'].rolling(window=240).mean()
        
        return df, None
    except Exception as e:
        return None, str(e)

def run_comprehensive_backtest(df, ki_pct, strike_pct, months):
    """綜合回測邏輯"""
    trading_days = int(months * 21)
    bt = df[['Date', 'Close']].copy()
    bt.columns = ['Start_Date', 'Start_Price']
    
    bt['End_Date'] = bt['Start_Date'].shift(-trading_days)
    bt['Final_Price'] = bt['Start_Price'].shift(-trading_days)
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=trading_days)
    bt['Min_Price_During'] = bt['Start_Price'].rolling(window=indexer, min_periods=1).min()
    
    bt = bt.dropna()
    
    if bt.empty: return None, None
    
    bt['KI_Level'] = bt['Start_Price'] * (ki_pct / 100)
    bt['Strike_Level'] = bt['Start_Price'] * (strike_pct / 100)
    
    bt['Touched_KI'] = bt['Min_Price_During'] < bt['KI_Level']
    bt['Below_Strike'] = bt['Final_Price'] < bt['Strike_Level']
    
    conditions = [
        (bt['Touched_KI'] == True) & (bt['Below_Strike'] == True),
        (bt['Touched_KI'] == True) & (bt['Below_Strike'] == False),
        (bt['Touched_KI'] == False)
    ]
    choices = ['Loss', 'Safe', 'Safe']
    bt['Result_Type'] = np.select(conditions, choices, default='Unknown')
    
    # 計算回本天數
    loss_indices = bt[bt['Result_Type'] == 'Loss'].index
    recovery_counts = [] 
    stuck_count = 0
    
    for idx in loss_indices:
        row = bt.loc[idx]
        target_price = row['Strike_Level']
        end_date = row['End_Date']
        future_data = df[(df['Date'] > end_date) & (df['Close'] >= target_price)]
        
        if not future_data.empty:
            days_needed = (future_data.iloc[0]['Date'] - end_date).days
            recovery_counts.append(days_needed)
        else:
            stuck_count += 1

    # Bar圖資料
    def calculate_bar_value(row):
        gap = ((row['Final_Price'] - row['Strike_Level']) / row['Strike_Level']) * 100
        return gap if row['Result_Type'] == 'Loss' else max(0, gap)

    bt['Bar_Value'] = bt.apply(calculate_bar_value, axis=1)
    bt['Color'] = np.where(bt['Result_Type'] == 'Loss', 'red', 'green')

    # 統計
    total = len(bt)
    safe_count = len(bt[bt['Result_Type'] == 'Safe'])
    safety_prob = (safe_count / total) * 100
    pos_count = len(bt[bt['Final_Price'] > bt['Start_Price']])
    pos_prob = (pos_count / total) * 100
    avg_recovery = np.mean(recovery_counts) if recovery_counts else 0
    
    stats = {
        'safety_prob': safety_prob,
        'positive_prob': pos_prob,
        'loss_count': len(loss_indices),
        'avg_recovery': avg_recovery,
        'stuck_count': stuck_count
    }
    
    return bt, stats

def plot_integrated_chart(df, ticker, current_price, p_ko, p_ki, p_st):
    """主圖：走勢 + 關鍵價位"""
    plot_df = df.tail(750).copy() # 顯示近3年
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], mode='lines', name='股價', line=dict(color='black', width=1.5)))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA20'], mode='lines', name='月線', line=dict(color='#3498db', width=1)))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA60'], mode='lines', name='季線', line=dict(color='#f1c40f', width=1)))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA240'], mode='lines', name='年線', line=dict(color='#9b59b6', width=1)))

    # KO
    fig.add_hline(y=p_ko, line_dash="dash", line_color="red", line_width=2)
    fig.add_annotation(x=1, y=p_ko, xref="paper", yref="y", text=f"KO: {p_ko:.2f}", showarrow=False, xanchor="left", font=dict(color="red"))
    # Strike
    fig.add_hline(y=p_st, line_dash="solid", line_color="green", line_width=2)
    fig.add_annotation(x=1, y=p_st, xref="paper", yref="y", text=f"Strike: {p_st:.2f}", showarrow=False, xanchor="left", font=dict(color="green"))
    # KI
    fig.add_hline(y=p_ki, line_dash="dot", line_color="orange", line_width=2)
    fig.add_annotation(x=1, y=p_ki, xref="paper", yref="y", text=f"KI: {p_ki:.2f}", showarrow=False, xanchor="left", font=dict(color="orange"))

    all_prices = [p_ko, p_ki, p_st, plot_df['Close'].max(), plot_df['Close'].min()]
    y_min, y_max = min(all_prices)*0.9, max(all_prices)*1.05

    fig.update_layout(title=f"{ticker} - 走勢與關鍵價位 (近3年)", height=450, margin=dict(r=80), xaxis_title="日期", yaxis_title="價格", yaxis_range=[y_min, y_max], hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    return fig

def plot_rolling_bar_chart(bt_data, ticker):
    """Bar 圖：回測結果"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bt_data['Start_Date'], y=bt_data['Bar_Value'], marker_color=bt_data['Color'], name='期末表現'))
    fig.add_hline(y=0, line_width=1, line_color="black")
    
    fig.update_layout(title=f"{ticker} - 滾動回測損益分佈 (2009至今)", xaxis_title="進場日期", yaxis_title="期末距離 Strike (%)", height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False, hovermode="x unified")
    return fig

# --- 4. 執行邏輯 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not ticker_list:
        st.warning("請輸入代碼")
    else:
        # 顯示當前選擇的 Issuer 資訊
        st.write(f"🏢 **當前分析發行機構 (Issuer):** `{issuer_input}`")
        
        for ticker in ticker_list:
            st.markdown(f"### 📌 標的：{ticker}")
            
            with st.spinner(f"正在分析 {ticker} (2009-Now) ..."):
                df, err = get_stock_data_from_2009(ticker)
            
            if err:
                st.error(f"{ticker} 讀取失敗: {err}")
                continue
                
            try:
                current_price = float(df['Close'].iloc[-1])
                p_ko = current_price * (ko_pct / 100)
                p_st = current_price * (strike_pct / 100)
                p_ki = current_price * (ki_pct / 100)
            except:
                st.error(f"{ticker} 價格計算錯誤")
                continue

            bt_data, stats = run_comprehensive_backtest(df, ki_pct, strike_pct, period_months)
            
            if bt_data is None:
                st.warning("資料不足")
                continue

            # ==========================================
            # B. 四大重點指標 (價位)
            # ==========================================
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新股價", f"{current_price:.2f}")
            c2.metric(f"KO ({ko_pct}%)", f"{p_ko:.2f}", help="若股價高於此，提前獲利出場")
            c3.metric(f"KI ({ki_pct}%)", f"{p_ki:.2f}", help="若股價跌破此，保護消失", delta_color="inverse")
            c4.metric(f"Strike ({strike_pct}%)", f"{p_st:.2f}", help="期初價格或接股成本")

            # ==========================================
            # C. 💰 潛在配息試算
            # ==========================================
            monthly_income = principal * (coupon_pa / 100) / 12
            
            st.markdown("#### 💰 潛在現金流試算 (Income Analysis)")
            m1, m2 = st.columns(2)
            m1.metric("投資本金", f"${principal:,.0f}")
            m2.metric("預估每月配息", f"${monthly_income:,.0f}", help=f"計算公式: 本金 x {coupon_pa}% / 12")
            st.divider()

            # ==========================================
            # D. 走勢及關鍵價位圖 (主圖)
            # ==========================================
            fig_main = plot_integrated_chart(df, ticker, current_price, p_ko, p_ki, p_st)
            st.plotly_chart(fig_main, use_container_width=True)

            # ==========================================
            # E. 藍底解釋 (AI 解讀)
            # ==========================================
            loss_pct = 100 - stats['safety_prob']
            stuck_rate = 0
            if stats['loss_count'] > 0:
                stuck_rate = (stats['stuck_count'] / stats['loss_count']) * 100
            avg_days = stats['avg_recovery']

            st.info(f"""
            **📊 長週期回測報告 (2009/01/01 至今，每 {period_months} 個月一期)：**
            
            1.  **獲利潛力 (正報酬機率)**：
                若不考慮配息，單純看股價，持有期滿後股價上漲的機率為 **{stats['positive_prob']:.1f}%**。
                
            2.  **安全性分析 (不被換到股票的機率)**：
                在過去 16 年任意時間點進場，有 **{stats['safety_prob']:.1f}%** 的機率可以安全拿回本金 (未跌破 KI 或 跌破後漲回)。
                
            3.  **恢復力分析 (回到 Strike 的時間)**：
                若不幸發生接股票的情況 (機率約 {loss_pct:.1f}%)，根據歷史經驗，**平均等待 {avg_days:.0f} 天** 股價即會漲回 Strike 價格。
                *(註：在所有接股票的案例中，約有 {stuck_rate:.1f}% 的情況截至目前尚未解套)*
            """)

            # ==========================================
            # F. 回測圖 (Bar Chart)
            # ==========================================
            st.subheader("📉 歷史滾動回測結果")
            st.caption("🟩 **綠色**：安全 (拿回本金) ｜ 🟥 **紅色**：接股票 (虧損幅度)")
            fig_bar = plot_rolling_bar_chart(bt_data, ticker)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

else:
    st.info("👈 請在左側設定參數，按下「開始分析」。")

# ==========================================
# 6. 底部警語
# ==========================================
st.markdown("""
<style>
.disclaimer-box {
    background-color: #fff3f3;
    border: 1px solid #e0b4b4;
    padding: 15px;
    border-radius: 5px;
    color: #8a1f1f;
    font-size: 0.9em;
    margin-top: 30px;
}
</style>
<div class='disclaimer-box'>
    <strong>⚠️ 免責聲明與投資風險預告</strong><br>
    1. <strong>本工具僅供教學與模擬試算</strong>：本系統計算之數據、圖表與機率僅供參考，不代表任何形式之投資建議，亦不保證未來獲利。<br>
    2. <strong>歷史不代表未來</strong>：回測數據基於 2009 年至今之歷史股價，過去的市場表現不保證未來的走勢。<br>
    3. <strong>非保本商品</strong>：結構型商品 (ELN/FCN) 為非保本型投資，最大風險為股價下跌導致本金全數虧損 (需承接價值減損之股票)。<br>
    4. <strong>實際條款為準</strong>：實際商品之觀察日、配息率、提前出場 (KO) 及敲入 (KI) 判定方式，請以發行機構之公開說明書及合約為準。<br>
    5. <strong>資料來源</strong>：股價資料來源為 Yahoo Finance 公開數據，可能存在延遲或誤差，本系統不保證資料之即時性與正確性。
</div>
""", unsafe_allow_html=True)
