"""
src/demo/app.py — Türkçe Dudak Okuma & En Sık 500 Kelime Avcısı Gradio Web Arayüzü
Kullanıcının webcam'den kaydettiği veya yüklediği sessiz videodan
dudak hareketlerini analiz ederek hedef 500 Türkçe kelimeyi zaman damgalarıyla yakalar.
"""

import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch

# Proje kök dizinini ekle
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vsr_conformer import VSRConformerModel
from src.spotter.keyword_spotter import KeywordSpotter
from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder
from src.vocab.top500_words import TOP_500_SET, TOP_500_WORDS, WORD_TO_RANK, get_word_viseme_signature
from src.vocab.turkish_vocab import VOCAB_SIZE, ctc_greedy_decode

# Cihaz seçimi (Apple Silicon MPS / CUDA / CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
CHECKPOINT_PATH = ROOT / "checkpoints" / "clean_vsr_best.pt"

# Global model ve çözücü tek seferlik önbelleği (Singleton)
_MODEL: Optional[VSRConformerModel] = None
_DECODER: Optional[LexiconBeamSearchDecoder] = None
_SPOTTER: Optional[KeywordSpotter] = None


def get_model_and_decoders():
    global _MODEL, _DECODER, _SPOTTER
    if _MODEL is None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint dosyası bulunamadı: {CHECKPOINT_PATH}\n"
                "Demo ancak full-training readiness kapısından geçen yeni kanonik "
                "model üretildikten sonra kullanılabilir."
            )
        model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=512, num_layers=3, encoder_type="bigru")
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        _MODEL = model

        _DECODER = LexiconBeamSearchDecoder(target_vocab=TOP_500_SET, beam_size=60, fps=25.0, viseme_tolerance=True)
        _SPOTTER = KeywordSpotter(target_vocab=TOP_500_SET, fps=25.0)

    return _MODEL, _DECODER, _SPOTTER


def preprocess_video(video_path: str) -> Tuple[torch.Tensor, float, int]:
    """
    Kullanıcı videosunu yükler, gri tonlamaya çevirir, 96x96'ya ölçekler/kırpar,
    88x88 merkez kırpma ve normalizasyon ile (1, 1, T, 88, 88) tensörü üretir.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video dosyası okunamadı: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Gri tonlama
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Boyutlandırma / Dudak Bölgesi Kırpma (96x96)
        h, w = gray.shape
        if h != 96 or w != 96:
            # Kullanıcı tüm yüz yüklediyse alt yarıyı (ağız bölgesi) odakla
            if h > 150 and w > 150:
                # Alt orta bölge (ağız/çene)
                mouth_y1 = int(h * 0.55)
                mouth_y2 = int(h * 0.95)
                mouth_x1 = int(w * 0.25)
                mouth_x2 = int(w * 0.75)
                gray_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                gray_96 = cv2.resize(gray_roi, (96, 96), interpolation=cv2.INTER_AREA)
            else:
                gray_96 = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        else:
            gray_96 = gray

        frames.append(gray_96)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Videodan hiçbir geçerli kare okunamadı!")

    arr = np.stack(frames).astype(np.float32) / 255.0  # (T, 96, 96) in [0.0, 1.0]
    T = arr.shape[0]

    # 88x88 merkez kırpma
    ch = (96 - 88) // 2
    cw = (96 - 88) // 2
    crop = arr[:, ch : ch + 88, cw : cw + 88]

    # Z-skor normalizasyonu
    normalized = (crop - 0.421) / 0.165
    tensor_in = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, T, 88, 88)
    return tensor_in, float(fps), T


def predict_lip_reading(
    video_path: Optional[str],
    blank_penalty: float = 1.0,
    beam_size: int = 50,
    viseme_tolerance: bool = True,
) -> Tuple[str, pd.DataFrame, str]:
    """Gradio tahmin boru hattı."""
    if not video_path:
        empty_df = pd.DataFrame(columns=["Kelime", "Başlangıç (sn)", "Bitiş (sn)", "Süre (ms)", "Sıra", "Güven", "Visem"])
        return "Lütfen bir video yükleyin veya webcam ile kaydedin.", empty_df, "Hazır"

    t_start = time.time()
    try:
        model, decoder, spotter = get_model_and_decoders()
        decoder.viseme_tolerance = viseme_tolerance

        # 1. Video Ön İşleme
        tensor_in, orig_fps, total_frames = preprocess_video(video_path)
        duration_sec = total_frames / orig_fps

        # 2. Model Çıkarımı
        with torch.no_grad():
            logits = model(tensor_in)[0]  # (T, Vocab)

        # 3. İki Kanallı Deşifreleme
        # A) CTC Greedy Decoding (Harf seviyesinde doğrudan gözlem)
        greedy_text = ctc_greedy_decode(logits.unsqueeze(0), blank_penalty=blank_penalty)[0]

        # B) Lexicon-Constrained CTC Prefix Beam Search (500 Kelimelik Sözlük & Visem İmzası)
        beam_text, detected_kws = decoder.decode(logits, beam_size=int(beam_size), blank_penalty=blank_penalty)

        # 4. Tablo Hazırlığı
        records = []
        for kw in detected_kws:
            dur_ms = int((kw.end_sec - kw.start_sec) * 1000)
            conf_pct = f"%{int(kw.confidence * 100)}"
            rank_str = f"#{kw.rank_in_500}" if kw.rank_in_500 != -1 else "-"
            viseme_sig = get_word_viseme_signature(kw.word) or "-"
            records.append({
                "Kelime": kw.word,
                "Başlangıç (sn)": f"{kw.start_sec:.2f}s",
                "Bitiş (sn)": f"{kw.end_sec:.2f}s",
                "Süre (ms)": f"{dur_ms} ms",
                "Sıra": rank_str,
                "Güven": conf_pct,
                "Visem": viseme_sig,
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=["Kelime", "Başlangıç (sn)", "Bitiş (sn)", "Süre (ms)", "Sıra", "Güven", "Visem"])

        elapsed = round(time.time() - t_start, 2)
        diag = (
            f"⏱️ Toplam Çıkarım Süresi: {elapsed}s | Kare Sayısı: {total_frames} ({duration_sec:.1f}s) | "
            f"Donanım: {DEVICE} | Ham Greedy CTC: '{greedy_text}'"
        )

        display_text = beam_text if beam_text.strip() else (f"(Ham CTC: {greedy_text})" if greedy_text.strip() else "Dudak hareketi algılanamadı.")
        return display_text, df, diag

    except Exception as e:
        empty_df = pd.DataFrame(columns=["Kelime", "Başlangıç (sn)", "Bitiş (sn)", "Süre (ms)", "Sıra", "Güven", "Visem"])
        return f"Hata oluştu: {str(e)}", empty_df, f"Hata: {str(e)}"


def build_app() -> gr.Blocks:
    """Gradio Arayüzünü Oluşturur."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    custom_css = """
    .hero-title {
        text-align: center;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(theme=theme, css=custom_css, title="Türkçe Dudak Okuma & 500 Kelime Avcısı") as demo:
        gr.HTML(
            """
            <div class="hero-title">
                <h1>👄 Türkçe Dudak Okuma & 500 Kelime Avcısı (VSR-TR-500)</h1>
            </div>
            <div class="hero-subtitle">
                Sessiz video akışından dudak hareketlerini analiz ederek Türkçedeki <b>en yaygın 500 kelimeyi</b> anlık yakalayan yapay zeka modeli.
            </div>
            <div class="badge-container">
                <span style="background: #e0e7ff; color: #3730a3; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">3D-ResNet18 + BiGRU</span>
                <span style="background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">Müfredat Eğitimi (Word ➔ Phrase)</span>
                <span style="background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">500 Kelimelik Sözlük & Visem Toleransı</span>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                video_input = gr.Video(
                    label="📹 Sessiz Video Girdisi (Webcam ile Konuşun veya Video Yükleyin)",
                    sources=["webcam", "upload"],
                    format="mp4",
                )

                with gr.Accordion("⚙️ Gelişmiş Çözücü Ayarları", open=False):
                    blank_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.5,
                        value=0.4,
                        step=0.1,
                        label="Boşluk Cezası (Blank Penalty)",
                        info="Yüksek değer harf ve kelime emisyonunu teşvik eder.",
                    )
                    beam_slider = gr.Slider(
                        minimum=10,
                        maximum=120,
                        value=50,
                        step=10,
                        label="Beam Arama Boyutu (Beam Size)",
                        info="Daha geniş beam arama sözlük doğruluğunu artırır.",
                    )
                    viseme_cb = gr.Checkbox(
                        value=True,
                        label="Visem / Homofen Toleransı",
                        info="Görsel olarak eşsesli dudak hareketlerini (b/p/m, f/v) otomatik eşler.",
                    )

                submit_btn = gr.Button("🔍 Dudakları Oku ve Kelimeleri Yakala", variant="primary", size="lg")

            with gr.Column(scale=6):
                transcript_out = gr.Textbox(
                    label="📝 Çözülen Cümle Transkripti (500 Kelimelik Türkçe Sözlük Kısıtlı)",
                    placeholder="Sonuç burada görüntülenecektir...",
                    lines=3,
                    interactive=False,
                )

                table_out = gr.DataFrame(
                    label="🎯 Yakalanan En Sık 500 Kelime (Zaman Damgaları & Güven Skorları)",
                    headers=["Kelime", "Başlangıç (sn)", "Bitiş (sn)", "Süre (ms)", "Sıra", "Güven", "Visem"],
                    interactive=False,
                )

                diagnostics_out = gr.Textbox(
                    label="ℹ️ Teşhis ve Çıkarım İstatistikleri",
                    interactive=False,
                    lines=2,
                )

        # Örnek Videolar Bölümü
        example_dir = ROOT / "data" / "master" / "-2uu72kg-9w"
        example_list = []
        if example_dir.exists():
            for p in sorted(example_dir.glob("*")):
                v_file = p / "mouth.mp4"
                if v_file.exists():
                    example_list.append([str(v_file), 1.0, 50, True])

        if example_list:
            gr.Examples(
                examples=example_list[:5],
                inputs=[video_input, blank_slider, beam_slider, viseme_cb],
                outputs=[transcript_out, table_out, diagnostics_out],
                fn=predict_lip_reading,
                cache_examples=False,
                label="💡 Örnek Test Klipleri (Tek Tıkla Deneyin)",
            )

        submit_btn.click(
            fn=predict_lip_reading,
            inputs=[video_input, blank_slider, beam_slider, viseme_cb],
            outputs=[transcript_out, table_out, diagnostics_out],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app = build_app()
    print(f"🚀 Türkçe Dudak Okuma Gradio Arayüzü Başlatılıyor: http://127.0.0.1:{port}")
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
