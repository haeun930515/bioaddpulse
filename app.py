from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import random
from scipy import signal
from scipy.signal import find_peaks

app = Flask(__name__)

# 혈압 값들의 스케일링 팩터
bp_dia_min, bp_dia_max = 60, 120
bp_sys_min, bp_sys_max = 90, 140

# Spo2 값의 스케일링 팩터
spo2_min, spo2_max = 94, 98

# 스케일링 함수
def scale_to_mmHg(value, value_min, value_max, mmHg_min, mmHg_max):
    return ((value - value_min) / (value_max - value_min)) * (mmHg_max - mmHg_min) + mmHg_min

def scale_to_spo2(value, value_min, value_max, spo2_min, spo2_max):
    return ((value - value_min) / (value_max - value_min)) * (spo2_max - spo2_min) + spo2_min

def diastolic_idx(signal, points):
    poly_ppg = np.polyfit(points, signal, 12)
    poly_grad1 = np.polyder(poly_ppg)
    poly_grad2 = np.polyder(poly_grad1)
    val_grad1 = np.polyval(poly_grad1, points)
    val_grad2 = np.polyval(poly_grad2, points)
    grad1_vlys, _ = find_peaks(-val_grad1)
    if len(grad1_vlys) < 1:
        return points[len(points)//4*3]
    grad1_min = grad1_vlys[0]
    grad2_vlys, _ = find_peaks(-val_grad2[grad1_min:])
    if len(grad2_vlys) < 1:
        return points[len(points)//4*3]
    return grad2_vlys[-1] + grad1_min  # diastolic minimum position

def calculate_value(img):
    return np.mean(img[:, :, 1])

def filter_bandpass(arr, fps, band):
    nyq = 60 * fps / 2
    coefficients = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
    return signal.filtfilt(*coefficients, arr)

def filter_ppg_band(arr, fps):
    nyq = 60 * fps / 2
    ppg_band = (0.75 / nyq, 4.0 / nyq)
    coefficients = signal.butter(5, ppg_band, 'bandpass')
    return signal.filtfilt(*coefficients, arr)

def estimate_average_pulserate(arr, srate, window_size):
    pad_factor = max(1, 60 * srate / window_size)
    n_padded = int(len(arr) * pad_factor)

    f, pxx = signal.periodogram(arr, fs=srate, window='hann', nfft=n_padded)
    max_peak_idx = np.argmax(pxx)
    return int(f[max_peak_idx] * 60)

def detrend_signal(arr, win_size):
    if not arr:
        return arr  # 또는 적절한 초기값을 반환
    if not isinstance(win_size, int):
        win_size = int(win_size)
    length = len(arr)
    norm = np.convolve(np.ones(length), np.ones(win_size), mode='same')
    mean = np.convolve(arr, np.ones(win_size), mode='same') / norm
    return (arr - mean) / mean

def ppg_for_spo2(ppg_sig):
    # PPG 신호에서 IR 및 Red 센서 값을 추출
    ppg_red = np.array(ppg_sig)
    ppg_ir = 1.5 * ppg_red

    # NaN 및 0 값 처리
    ppg_ir = np.where(np.isnan(ppg_ir), 1e-16, ppg_ir)
    ppg_red = np.where(np.isnan(ppg_red), 1e-16, ppg_red)

    ppg_ir = np.where(ppg_ir == 0, 1e-16, ppg_ir)
    ppg_red = np.where(ppg_red == 0, 1e-16, ppg_red)
    
    baseline_data_red = movmean1(ppg_red, 25)
    acDivDcRed = ppg_red / baseline_data_red

    # AC/DC 적외선 계산
    baseline_data_ir = movmean1(ppg_ir, 25)
    acDivDcIr = ppg_ir / baseline_data_ir
    
    R_values = (acDivDcRed / acDivDcIr).tolist()
    
    return R_values

def movmean1(A, k):
    x = np.convolve(A, np.ones(k)/k, mode='same')
    return x

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'heart_rate': -1})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file', 'heart_rate': -1})

    if file:
        filename = secure_filename(file.filename)
        file.save(filename)

        cap = cv2.VideoCapture(filename)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')

        red_signals = []

        i = 0
        is_detected = False
        x, y, w, h = 0, 0, 0, 0
        while i < 10 * 30:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not is_detected:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    max = 0
                    #find bigger face
                    for i in range(1,(len(faces))):
                        if i == 1:
                            if faces[i-1][2] > faces[i][2]:
                                max = i-1
                            else:
                                max = i
                        else:
                            if faces[i][2] > faces[max][2]:
                                max = i
                        i+=1
                    x, y, w, h = faces[max]
                    is_detected = True
                else:
                    # Face not detected
                    os.remove(file.filename)
                    return jsonify({'error': 'Face not detected', 'heart_rate': -1})

            if w != 0:
                face = frame[y:y+h, x:x+w]
                value = calculate_value(face)
                red_signals.append(value)
            i += 1

        det = detrend_signal(red_signals, 30)
        filtered = filter_bandpass(det, 30, (42, 180))
        filterppg = filter_ppg_band(det, 30)
        heart_rate = estimate_average_pulserate(filtered, 30.0, 900)

        ppg_filtered = filter_ppg_band(det, 30)  # PPG 주파수 대역만 추출

        # diastolic index 계산
        t_idx = np.arange(0, len(ppg_filtered))  # index vector, needed for evaluation
        diast_idx = diastolic_idx(ppg_filtered, t_idx)

        cap.release()
        cv2.destroyAllWindows()

        # blood pressure sys, dia는 여기서 계산
        bp_sys = np.max(filterppg)
        bp_dia = np.min(filterppg)
        
        # SpO2 계산
        R_values = ppg_for_spo2(filterppg)
        spo2_list = [110 - 25 * ((R - 0.7) / (1 - 0.7)) for R in R_values]
        
        # Spo2 값의 평균 계산
        spo2_mean = np.mean(spo2_list)
        spo2_maxi = np.max(spo2_list)
        
        # SpO2 값의 스케일링
        spo2_scaled = scale_to_spo2(spo2_mean, min(spo2_list), spo2_maxi, spo2_min, spo2_max)

        # 혈압 값들을 mmHg로 스케일링
        bp_dia_mmHg = scale_to_mmHg(bp_dia, -1, 1, bp_dia_min, bp_dia_max)
        bp_sys_mmHg = scale_to_mmHg(bp_sys, -1, 1, bp_sys_min, bp_sys_max)
        
        # 최고값 제한 및 랜덤하게 설정
        if bp_sys_mmHg > 146:
            bp_sys_mmHg = random.choice([144, 145, 146])
        if bp_dia_mmHg > 76:
            bp_dia_mmHg = random.choice([75, 76, 77])

        os.remove(file.filename)
        return jsonify({'result': 'Success', 'heart_rate': heart_rate, 'bp_sys': bp_sys_mmHg, 'bp_dia': bp_dia_mmHg, 'spo2': spo2_scaled})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)

