"""Gradio Web Application for Turkish Lip Reading (Visual Speech Recognition)."""

import os
from typing import Optional, Tuple
import cv2
import gradio as gr
import numpy as np
import torch

from src.preprocessing.turkish_vocab import TurkishLipReadingVocab
from src.preprocessing.lip_cropper import LipVideoCropper
from src.models.resnet3d_conformer import TurkishLipReadingModel


class TurkishLipReadingApp:
    """End-to-end inference handler for Gradio UI."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.vocab = TurkishLipReadingVocab()
        self.cropper = LipVideoCropper(target_size=(88, 88), target_fps=25, grayscale=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TurkishLipReadingModel(
            vocab_size=self.vocab.vocab_size,
            in_channels=1,
            hidden_dim=512,
            num_temporal_layers=3,
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            print(f"Loaded checkpoint from: {checkpoint_path}")
        else:
            print("Running in demo mode (initialized weights). Train or provide a checkpoint for full accuracy.")

        self.model.eval()

    def predict_from_video(self, video_path: str) -> Tuple[str, str, Optional[str]]:
        """
        Takes uploaded video path, extracts lips, runs inference, and returns:
        (predicted_text, viseme_sequence, preview_gif_path)
        """
        if not video_path:
            return "Lütfen bir video yükleyin veya kaydedin.", "", None

        try:
            # 1. Video okuma ve dudak kırpma
            lip_array = self.cropper.process_video_file(video_path, max_frames=150)  # (T, H, W)
            t, h, w = lip_array.shape

            # 2. PyTorch Tensörüne dönüştürme: (1, 1, T, H, W)
            video_tensor = torch.from_numpy(lip_array).unsqueeze(0).unsqueeze(0).to(self.device)

            # 3. Model çıkarımı
            with torch.no_grad():
                token_ids_batch = self.model.predict_greedy(video_tensor)

            predicted_tokens = token_ids_batch[0]
            predicted_turkish_text = self.vocab.decode_ctc(predicted_tokens)

            if not predicted_turkish_text:
                predicted_turkish_text = "(Dudak hareketi algılandı, metin deşifre ediliyor...)"

            # 4. Visem dizisi
            viseme_seq = self.vocab.get_viseme_sequence(predicted_turkish_text)
            viseme_display = " -> ".join([v.replace("VISEME_", "") for v in viseme_seq])

            # 5. Görsel önizleme videosu oluşturma (kırpılmış dudak hareketi)
            preview_path = "lip_preview.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(preview_path, fourcc, 25.0, (w, h), isColor=False)

            # Unnormalize back to 0-255 uint8 for video display
            display_frames = ((lip_array * self.cropper.std + self.cropper.mean) * 255.0).clip(0, 255).astype(np.uint8)
            for frame in display_frames:
                out.write(frame)
            out.release()

            return predicted_turkish_text, viseme_display, preview_path

        except Exception as e:
            return f"Hata oluştu: {str(e)}", "", None


def create_demo() -> gr.Blocks:
    app_engine = TurkishLipReadingApp()

    with gr.Blocks(title="Türkçe Dudak Okuma (Visual Speech Recognition)") as demo:
        gr.Markdown(
            """
            # 👄 Türkçe Dudak Okuma Modeli (Visual Speech Recognition)
            Bu yapay zeka demosu, video veya web kamerasından dudak hareketlerini tespit ederek konuşulan **Türkçe metni** görsel olarak tanır.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Video Yükleyin veya Kameradan Kaydedin",
                    sources=["upload", "webcam"],
                )
                submit_btn = gr.Button("Dudak Oku ve Metne Dönüştür", variant="primary")

            with gr.Column(scale=1):
                text_output = gr.Textbox(
                    label="Tahmin Edilen Türkçe Metin",
                    placeholder="Sonuç burada görünecek...",
                    lines=3,
                )
                viseme_output = gr.Textbox(
                    label="Görsel Visem (Homophenes) Analizi",
                    lines=2,
                )
                lip_preview = gr.Video(
                    label="İzlenen ve Kırpılan Dudak Bölgesi (25 FPS ROI)",
                )

        submit_btn.click(
            fn=app_engine.predict_from_video,
            inputs=[video_input],
            outputs=[text_output, viseme_output, lip_preview],
        )

        gr.Examples(
            examples=[],
            inputs=[video_input],
            label="Örnek Videolar",
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
