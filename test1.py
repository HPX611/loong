import numpy as np
import matplotlib.pyplot as plt
import pywt

# 从txt文件加载ECG数据
file_path = r'F:\loong\training_dataset\S1-N-301.txt'
ecg_data = np.loadtxt(file_path)

# 创建时间序列（假设采样频率为每秒250个采样点）
sampling_frequency = 250  # 采样频率
time = np.arange(len(ecg_data)) / sampling_frequency

# 小波变换去噪
def denoise_ecg(ecg_data):
    # 选择小波函数和阈值类型
    wavelet = 'db4'  # 选取 Daubechies 4 小波
    threshold_mode = 'soft'  # 软阈值

    # 小波变换
    coeffs = pywt.wavedec(ecg_data, wavelet)

    # 对每个细节系数进行阈值处理
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(ecg_data)))
    coeffs = [pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs]

    # 重构信号
    denoised_ecg = pywt.waverec(coeffs, wavelet)
    return denoised_ecg

# 对ECG信号进行去噪
denoised_ecg_data = denoise_ecg(ecg_data)

# 绘制原始ECG信号和去噪后的信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, ecg_data, color='blue')
plt.title('Original ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, denoised_ecg_data, color='red')
plt.title('Denoised ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
