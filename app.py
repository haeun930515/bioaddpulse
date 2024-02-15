from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from scipy import signal

app = Flask(__name__)

band = (42, 180)
win_size = 900

def calculate_value(img):
    return np.mean(img[:, :, 1])

def filter_bandpass(arr, fps, band):
    nyq = 60 * fps / 2
    coefficients = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
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
                    max_face_index = np.argmax(faces[:, 2])  # Index of the face with maximum width
                    x, y, w, h = faces[max_face_index]
                    is_detected = True
                else:
                    # Face not detected
                    return jsonify({'error': 'Face not detected', 'heart_rate': -1})

            if w != 0:
                face = frame[y:y+h, x:x+w]
                value = calculate_value(face)
                red_signals.append(value)
            i += 1

        det = detrend_signal(red_signals, 30)
        filtered = filter_bandpass(det, 30, (42, 180))
        heart_rate = estimate_average_pulserate(filtered, 30.0, 900)

        cap.release()
        cv2.destroyAllWindows()

        return jsonify({'result': 'Success', 'heart_rate': heart_rate})

if __name__ == '__main__':
    app.run(debug=True)
