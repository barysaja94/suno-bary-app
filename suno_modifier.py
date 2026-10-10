import gradio as gr
import os
import time
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

# --- FUNGSI FILTER DSP ---
def butter_bandpass_filter(data, lowcut=130.0, highcut=9500.0, fs=44100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- FUNGSI MODIFIKASI AUDIO ---
def process_suno_anti_detection(vocal_path, pitch_steps, speed_rate, apply_filter, convert_mono):
    if not vocal_path: return None
    try:
        y, audio_sr = librosa.load(vocal_path, sr=44100, mono=convert_mono)
        y, _ = librosa.effects.trim(y, top_db=25)
        
        if pitch_steps != 0:
            y = librosa.effects.pitch_shift(y, sr=audio_sr, n_steps=float(pitch_steps))
        if speed_rate != 1.0:
            y = librosa.effects.time_stretch(y, rate=float(speed_rate))
            
        if apply_filter:
            if y.ndim == 1:
                y = butter_bandpass_filter(y, fs=audio_sr)
            else:
                y[0] = butter_bandpass_filter(y[0], fs=audio_sr)
                y[1] = butter_bandpass_filter(y[1], fs=audio_sr)
        
        y = librosa.util.normalize(y)
        os.makedirs("processed_samples", exist_ok=True)
        out_path = os.path.join("processed_samples", f"suno_bypass_{int(time.time())}.wav")
        sf.write(out_path, y if y.ndim == 1 else y.T, audio_sr)
        return out_path
    except Exception as e:
        print(f"Error Processing Audio: {e}")
        return None

# --- DESAIN TAMPILAN TEMA GELAP ---
custom_css = """
body, .gradio-container { background-color: #0b0e14 !important; color: #ffffff !important; font-family: 'Segoe UI', sans-serif !important; }
.header-box { background: #121824; border: 1px solid #ff9100; border-left: 6px solid #ff9100; border-radius: 8px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(255, 145, 0, 0.1); }
.panel-area { background-color: #1a233b !important; border: 1px solid #1f293d !important; border-radius: 12px !important; padding: 25px !important; }
.btn-proses { background: linear-gradient(90deg, #ff6d00 0%, #ff9100 100%) !important; color: #ffffff !important; font-weight: 800 !important; font-size: 18px !important; border: none !important; border-radius: 8px !important; padding: 15px !important; margin-top: 20px !important; }
.block, .file-drop, div[data-testid="audio"] { background-color: #121824 !important; border: 1px dashed #ff9100 !important; }
"""

# --- MEMBANGUN ANTARMUKA ---
with gr.Blocks(css=custom_css, title="Suno Modifier") as app:
    gr.HTML("""
    <div class="header-box">
        <h1 style="color:#ffffff; font-size:26px; font-weight:800; margin:0 0 5px 0;">🎛️ SUNO SAMPLE MODIFIER</h1>
        <p style="color:#ff9100; margin:0;">Bypass AI Copyright Fingerprinting (Pitch, Speed, EQ Filter)</p>
    </div>
    """)
    
    with gr.Column(elem_classes="panel-area"):
        vocal_input = gr.Audio(type="filepath", label="📂 Upload Audio Vokal (WAV/MP3)")
        
        with gr.Row():
            pitch = gr.Slider(-5, 5, step=1, value=2, label="🎛️ Pitch Shift (Semitone)")
            speed = gr.Slider(0.8, 1.2, step=0.05, value=1.05, label="⚡ Speed Rate")
            
        with gr.Row():
            filter_check = gr.Checkbox(True, label="🧹 Bandpass EQ Filter")
            mono_check = gr.Checkbox(True, label="🎧 Convert to Mono")
            
        btn = gr.Button("🔥 PROSES MODIFIKASI", elem_classes="btn-proses")
        out = gr.Audio(label="🔥 HASIL VOKAL BYPASS")
        
        btn.click(process_suno_anti_detection, inputs=[vocal_input, pitch, speed, filter_check, mono_check], outputs=out)

if __name__ == "__main__":
    print("Mengaktifkan Server Suno Modifier (Cloud Mode)...")
    app.launch(server_name="0.0.0.0")