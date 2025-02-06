import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title("上包絡偵測程式")  # 應用程式標題

# 上傳 CSV 檔案
uploaded_file = st.file_uploader("上傳您的 CSV 檔案", type=["csv"])  # 上傳檔案

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### 上傳的資料")  # 顯示上傳的資料標題
        st.dataframe(data)  # 顯示資料表
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤: {e}")
        st.stop()

    # 使用者輸入 CSV 欄位名稱
    a = st.text_input('輸入 CSV 檔的 **時間欄位名稱**:', value=data.columns[0])
    b = st.text_input('輸入 CSV 檔的 **位移欄位名稱**:', value=data.columns[1])

    if a not in data.columns or b not in data.columns:
        st.error("請輸入正確的欄位名稱！")
        st.stop()

    time = data[a]  # 時間欄位
    amplitude = data[b]  # 振幅欄位

    # 峰值檢測
    peaks, _ = find_peaks(amplitude)
    if len(peaks) == 0:
        st.warning("未檢測到任何峰值，請檢查數據！")
        st.stop()
    
    peak_times = time.iloc[peaks].values.reshape(-1, 1)  # 峰值時間
    peak_amplitudes = amplitude.iloc[peaks]  # 峰值振幅
    log_peak_amplitudes = np.log(peak_amplitudes)  # 計算峰值振幅的自然對數

    # 繪製 振幅 vs 時間 圖
    st.write("### 振幅 vs 時間")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time, amplitude, label='Damped Oscillation', color='blue')  # 繪製原始信號
    ax1.plot(peak_times.flatten(), peak_amplitudes, 'ro', label='Detected Peaks')  # 標記峰值
    ax1.set_title('Amplitude vs Time', fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Amplitude (m)', fontsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid()
    st.pyplot(fig1)

    # 執行線性回歸
    model = LinearRegression()
    model.fit(peak_times, log_peak_amplitudes)  # 訓練回歸模型
    slope = model.coef_[0]  # 斜率
    intercept = model.intercept_  # 截距
    log_amplitude_pred = model.predict(peak_times)  # 預測值
    r_squared = model.score(peak_times, log_peak_amplitudes)  # R 平方值

    # 繪製 ln(峰值振幅) vs 時間 圖
    st.write("### ln(峰值振幅) vs 時間")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(peak_times, log_peak_amplitudes, color='red', label='Data (ln-transformed)')  # 資料點
    ax2.plot(peak_times, log_amplitude_pred, 
             label=f'Fit: ln(A) = {slope:.5f}t + {intercept:.5f}')  # 回歸線
    ax2.set_title('Ln of Peak Amplitudes vs Time', fontsize=16)
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('ln(Peak Amplitude)', fontsize=14)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid()

    # 顯示回歸方程和 R 平方值
    text_x = peak_times.mean()
    text_y = log_amplitude_pred.mean()
    ax2.text(text_x + (peak_times.max() - peak_times.min()) * 0.1, 
             text_y, 
             f'ln(A) = {slope:.5f}t + {intercept:.5f}\n$R^2$ = {r_squared:.4f}', 
             fontsize=12)
    st.pyplot(fig2)

    # 顯示回歸結果
    st.write("### 回歸結果")
    st.write(f"斜率 (衰減率): {slope:.5f}")
    st.write(f"截距: {intercept:.5f}")
    st.write(f"R² 值: {r_squared:.4f}")

