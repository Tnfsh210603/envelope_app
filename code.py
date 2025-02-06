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
    # 讀取資料
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
    try:
        peaks, _ = find_peaks(amplitude)
        if len(peaks) == 0:
            st.warning("未檢測到任何峰值，請檢查數據！")
            st.stop()
        
        peak_times = time.iloc[peaks].values.reshape(-1, 1)  # 峰值時間
        peak_amplitudes = amplitude.iloc[peaks]  # 峰值振幅
        log_peak_amplitudes = np.log(peak_amplitudes)  # 計算峰值振幅的自然對數
    except Exception as e:
        st.error(f"峰值檢測錯誤: {e}")
        st.stop()

    # 建立峰值資料的 DataFrame
    peak_data = pd.DataFrame({
        '峰值時間': peak_times.flatten(),
        '峰值振幅': peak_amplitudes,
        'ln(峰值振幅)': log_peak_amplitudes
    })

    # 顯示峰值資料
    st.write("### 峰值數據（含自然對數）")  # 顯示表格標題
    st.dataframe(peak_data)  # 顯示資料表

    # 繪製 振幅 vs 時間 圖
    st.write("### 上包絡 vs 時間")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time, amplitude, label='Displacement')  # 繪製原始信號
    ax1.plot(peak_times.flatten(), peak_amplitudes, 'ro', label='Envelope')  # 標記峰值
    ax1.set_title('Displacement vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Displacement (m)')
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

    # 執行線性回歸
    try:
        model = LinearRegression()
        model.fit(peak_times, log_peak_amplitudes)  # 訓練回歸模型
        slope = model.coef_[0]  # 斜率
        intercept = model.intercept_  # 截距
        log_amplitude_pred = model.predict(peak_times)  # 預測值
        r_squared = model.score(peak_times, log_peak_amplitudes)  # R 平方值
    except Exception as e:
        st.error(f"回歸分析時發生錯誤: {e}")
        st.stop()

    # 繪製 ln(峰值振幅) vs 時間 圖
    st.write("### ln(上包絡位移) vs 時間")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(peak_times, log_peak_amplitudes, color='red', label='ln transformed data')  # 資料點
    ax2.plot(peak_times, log_amplitude_pred, color='blue',
             label=f'Fit: ln(A) = {slope:.4f}t + {intercept:.4f}')  # 回歸線
    ax2.set_title('ln(Displacement) vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('ln(Displacement)')
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

    # 顯示回歸結果
    st.write("### 回歸結果")
    st.write(f"斜率 (衰減率): {slope:.5f}")
    st.write(f"截距: {intercept:.5f}")
    st.write(f"R² 值: {r_squared:.4f}")

