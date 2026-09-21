"""
src/data/dataset.py — Türkçe Dudak Okuma PyTorch Veri Kümesi ve Veri Artırma
"""

import json
import math
import os
import pathlib
import random
import subprocess
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.vocab.turkish_vocab import (
    BLANK_IDX,
    PAD_IDX,
    normalize_turkish_text,
    text_to_indices,
)
from src.vocab.top500_words import TOP_500_SET

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_allowed_videos(
    master_path: pathlib.Path,
    split: Optional[str],
    split_map_path: Optional[Union[str, pathlib.Path]],
) -> Optional[Set[str]]:
    """Split istendiğinde haritayı zorunlu tutar; haritasız tam veri kullanımını engeller."""
    if not split:
        return None

    if split_map_path is not None:
        candidates = [pathlib.Path(split_map_path)]
    elif "iborotti" in str(master_path).lower():
        candidates = [
            ROOT / "data" / "metadata" / "split_map_iborotti.json",
            pathlib.Path(__file__).resolve().parent / "split_map_iborotti.json",
            pathlib.Path("/root/data/metadata/split_map_iborotti.json"),
        ]
    else:
        candidates = [
            ROOT / "data" / "metadata" / "split_map.json",
            pathlib.Path("/root/data/metadata/split_map.json"),
        ]
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    if existing is None:
        if split_map_path is None and "iborotti" in str(master_path).lower():
            from src.data.split_map_data import IBOROTTI_SPLIT_MAP
            split_map = IBOROTTI_SPLIT_MAP
        else:
            searched = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(
                f"Split haritası bulunamadı (split={split}). Aranan yollar: {searched}"
            )
    else:
        try:
            with open(existing, "r", encoding="utf-8") as handle:
                split_map = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Split haritası okunamadı: {existing}: {exc}") from exc

    allowed = {video_id for video_id, info in split_map.items() if info.get("split") == split}
    if not allowed:
        raise ValueError(f"Split haritasında '{split}' için video bulunamadı: {existing}")
    return allowed


def _metadata_is_accepted(meta_file: pathlib.Path, require_acceptance: bool) -> bool:
    """Açıkça reddedilmiş veya HF kaynağında kabulü kanıtlanmamış klipleri dışlar."""
    if not meta_file.exists():
        return not require_acceptance
    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    accepted = metadata.get("accepted")
    quality_status = str(metadata.get("quality_status", "")).lower()
    visual_status = str((metadata.get("visual_quality") or {}).get("status", "")).lower()
    if accepted is False or quality_status == "rejected" or visual_status == "rejected":
        return False
    if require_acceptance:
        return accepted is True and quality_status == "accepted" and visual_status == "accepted"
    return True



def load_video_frames(video_path: Union[str, pathlib.Path]) -> np.ndarray:
    """
    96x96 gri tonlama videoyu (T, H, W) float32 numpy dizisi olarak yükler.
    Önce PyAV (av) kütüphanesini dener, başarısız olursa ffmpeg boru hattı ile yükler.
    """
    path_str = str(video_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Video dosyası bulunamadı: {path_str}")

    frames = []
    # 1. PyAV ile hızlı doğrudan C-FFmpeg çözümü
    try:
        import av
        container = av.open(path_str)
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            gray = frame.to_ndarray(format="gray")
            frames.append(gray)
        container.close()
    except Exception:
        frames = []

    # 2. PyAV başarısız olursa ffmpeg boru hattı (pipe)
    if not frames:
        cmd = [
            "ffmpeg",
            "-i", path_str,
            "-vf", "scale=96:96",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-v", "error",
            "-"
        ]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw, _ = p.communicate()
            if p.returncode == 0 and len(raw) > 0:
                frame_size = 96 * 96
                n_frames = len(raw) // frame_size
                arr = np.frombuffer(raw[:n_frames * frame_size], dtype=np.uint8)
                frames = list(arr.reshape((n_frames, 96, 96)))
        except Exception as e:
            raise RuntimeError(f"Video ffmpeg ile de açılamadı [{path_str}]: {e}")

    if len(frames) == 0:
        raise ValueError(f"Videodan hiçbir kare okunamadı: {path_str}")

    arr = np.stack(frames).astype(np.float32) / 255.0  # (T, 96, 96) in [0.0, 1.0]
    return arr


class LipReadingDataset(Dataset):
    """
    Türkçe Dudak Okuma Veri Kümesi.
    - mouth.mp4: (T, 96, 96) gri tonlamalı dudak bölgesi.
    - align.json: Transkript ve kelime zaman damgaları.
    - Veri artırma: Random Crop (88x88), Yatay Çevirme (p=0.5), Zamansal Maskeleme (Time Masking).
    """

    def __init__(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        master_dir: Optional[Union[str, pathlib.Path]] = None,
        data_dir: Optional[Union[str, pathlib.Path]] = None,
        is_train: bool = True,
        split: Optional[str] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        crop_size: int = 88,
        max_frames: int = 250,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        apply_horizontal_flip: bool = True,
        apply_time_masking: bool = True,
        apply_temporal_jitter: bool = True,
        apply_photometric_jitter: bool = True,
        cache_in_ram: bool = False,
    ):
        self.is_train = is_train
        self.crop_size = crop_size
        if max_duration is not None:
            calc_max_frames = int(math.ceil(max_duration * 25.0))
            self.max_frames = max(max_frames, calc_max_frames)
        else:
            self.max_frames = max(max_frames, 500)
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.apply_horizontal_flip = apply_horizontal_flip and is_train
        self.apply_time_masking = apply_time_masking and is_train
        self.apply_temporal_jitter = apply_temporal_jitter and is_train
        self.apply_photometric_jitter = apply_photometric_jitter and is_train
        self.cache_in_ram = cache_in_ram

        self.samples: List[Dict[str, Any]] = []

        if samples is not None:
            self.samples = samples
        else:
            scan_dir = master_dir or data_dir
            if scan_dir is None:
                candidates = [
                    ROOT / "data" / "iborotti",
                    ROOT / "data" / "master",
                    pathlib.Path("/root/data/iborotti"),
                    pathlib.Path("/root/data/master"),
                ]
                for c in candidates:
                    if c.exists() and (len(list(c.glob("*/*/mouth.mp4"))) > 0 or len(list(c.glob("clips/*/*/mouth.mp4"))) > 0):
                        scan_dir = c
                        break

            if scan_dir is not None:
                self.samples = self._scan_master_dir(
                    pathlib.Path(scan_dir),
                    split=split,
                    split_map_path=split_map_path,
                    max_duration=self.max_duration,
                    min_duration=self.min_duration,
                )

        # Bellek İçi Önbellekleme (In-Memory RAM Cache)
        # Disk I/O ve FFmpeg/PyAV deşifreleme darboğazını ortadan kaldırır (30x-50x hızlanma)
        self._cached_frames: Optional[List[np.ndarray]] = None
        if self.cache_in_ram and len(self.samples) > 0:
            self._preload_into_ram()

    @staticmethod
    def _scan_master_dir(
        master_path: pathlib.Path,
        split: Optional[str] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """data/iborotti/ veya data/master/ dizinini tarayarak geçerli video ve transkript çiftlerini toplar."""
        collected = []
        if not master_path.exists():
            return collected

        allowed_videos = _load_allowed_videos(master_path, split, split_map_path)

        search_roots = [master_path]
        if (master_path / "clips").exists():
            search_roots.append(master_path / "clips")

        found_mouths = []
        for s_root in search_roots:
            found_mouths.extend(sorted(s_root.glob("*/*/mouth.mp4")))
        found_mouths = sorted(list(set(found_mouths)))

        for mouth_file in found_mouths:
            seg_dir = mouth_file.parent
            vid = seg_dir.parent.name
            seg_id = seg_dir.name
            if allowed_videos is not None and vid not in allowed_videos:
                continue

            text = ""
            duration = 0.0
            text_file = seg_dir / "transcript.txt"
            align_file = seg_dir / "align.json"
            meta_file = seg_dir / "metadata.json"

            require_acceptance = "iborotti" in str(master_path).lower()
            if not _metadata_is_accepted(meta_file, require_acceptance=require_acceptance):
                continue

            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    duration = float(m_data.get("duration", 0.0))
                    text = m_data.get("text", "").strip()
                except Exception:
                    pass

            if text_file.exists():
                try:
                    with open(text_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                except Exception:
                    pass
            elif align_file.exists() and not text:
                try:
                    with open(align_file, "r", encoding="utf-8") as f:
                        align_data = json.load(f)
                    text = align_data.get("text", "").strip()
                except Exception:
                    pass

            if not text:
                continue

            # Seviye 1 Veri Kalitesi: Altyazı jeneriği ve tekil harf gürültüsünü ele
            text_lower = text.lower()
            if "altyazı" in text_lower or len(text.split()) < 2:
                continue

            # Süre filtresi (Utterance length curriculum)
            if max_duration is not None and duration > 0 and duration > max_duration:
                continue
            if min_duration is not None and duration > 0 and duration < min_duration:
                continue

            collected.append({
                "video_path": str(mouth_file),
                "align_path": str(align_file) if align_file.exists() else (str(meta_file) if meta_file.exists() else None),
                "transcript": text,
                "duration": duration,
                "video_id": vid,
                "seg_id": seg_id,
            })
        return collected

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, video: np.ndarray, target_len: int = 0) -> np.ndarray:
        """
        video: (T, H, W) float32
        Dönüş: (T, crop_size, crop_size) float32
        """
        T, H, W = video.shape

        # Boyut crop_size'dan küçükse pad yap
        if H < self.crop_size or W < self.crop_size:
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            video = np.pad(video, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            T, H, W = video.shape

        # 1. Kırpma (Random veya Center Crop)
        if self.is_train:
            max_y = max(0, H - self.crop_size)
            max_x = max(0, W - self.crop_size)
            top = random.randint(0, max_y) if max_y > 0 else 0
            left = random.randint(0, max_x) if max_x > 0 else 0
        else:
            top = (H - self.crop_size) // 2
            left = (W - self.crop_size) // 2

        video = video[:, top : top + self.crop_size, left : left + self.crop_size]

        # 2. Yatay Çevirme (Horizontal Flip)
        if self.apply_horizontal_flip and random.random() < 0.5:
            video = np.flip(video, axis=-1).copy()

        # 3. Zamansal Maskeleme (Time Masking)
        if self.apply_time_masking and random.random() < 0.5 and T > 8:
            max_mask_len = max(2, int(0.15 * T))
            mask_len = random.randint(1, max_mask_len)
            mask_start = random.randint(0, T - mask_len)
            # Ortalama gri ton ile maskele
            video[mask_start : mask_start + mask_len] = np.mean(video)

        # 4. Zamansal Hız Pertürbasyonu ve Kare Seyreltme (Speed Perturbation / Temporal Jitter)
        min_required_T = max(8, target_len + 2)
        if self.apply_temporal_jitter and random.random() < 0.4 and T > min_required_T + 4:
            mode = random.choice(["speed", "dropout"])
            if mode == "speed":
                # 0.90x ile 1.10x arası hız değişimi (hızlı / yavaş dudak devinimini modelleme)
                speed_factor = random.uniform(0.90, 1.10)
                new_T = max(min_required_T, int(round(T * speed_factor)))
                indices = np.round(np.linspace(0, T - 1, new_T)).astype(int)
                indices = np.clip(indices, 0, T - 1)
                video = video[indices]
                T = video.shape[0]
            elif mode == "dropout":
                # 1-2 ardışık olmayan kareyi düşürerek (dropout) zaman içi dayanıklılık sağlama
                max_drop = min(2, (T - min_required_T) // 2)
                if max_drop >= 1:
                    n_drop = random.randint(1, max_drop)
                    drop_indices = set(random.sample(range(1, T - 1), n_drop))
                    keep_indices = [i for i in range(T) if i not in drop_indices]
                    video = video[keep_indices]
                    T = video.shape[0]

        # 5. Fotometrik Parlaklık ve Kontrast Çeşitlemesi (Cross-Speaker Lighting Invariance)
        if self.apply_photometric_jitter and random.random() < 0.6:
            alpha = random.uniform(0.85, 1.15)  # Kontrast
            beta = random.uniform(-0.10, 0.10)  # Parlaklık
            video = np.clip(video * alpha + beta, 0.0, 1.0)

        return video

    def _preload_into_ram(self) -> None:
        """Tüm video karelerini RAM'e uint8 olarak önbelleğe alır (Disk I/O darboğazını yok eder)."""
        import time
        t0 = time.time()
        print(f"📦 [{len(self.samples)} Segment] RAM'e önbelleğe alınıyor...")
        self._cached_frames = []
        for i, item in enumerate(self.samples):
            try:
                raw_float = load_video_frames(item["video_path"])
                raw_uint8 = (raw_float * 255.0).clip(0, 255).astype(np.uint8)
                self._cached_frames.append(raw_uint8)
            except Exception as e:
                # Bozuk dosya durumunda boş dizi ekle
                self._cached_frames.append(np.zeros((1, 96, 96), dtype=np.uint8))
        elapsed = round(time.time() - t0, 2)
        total_mb = sum(arr.nbytes for arr in self._cached_frames) / (1024 * 1024)
        print(f"✅ Önbellekleme Tamamlandı: {total_mb:.1f} MB RAM tahsis edildi ({elapsed}s, ~{1000*elapsed/max(1, len(self.samples)):.1f} ms/video).")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        transcript = item["transcript"]

        clean_text = normalize_turkish_text(transcript)
        token_indices = text_to_indices(clean_text)
        target_tensor = torch.tensor(token_indices, dtype=torch.long)

        # Video yükleme (RAM önbelleğinden veya diskten)
        if self._cached_frames is not None and idx < len(self._cached_frames):
            video_arr = self._cached_frames[idx].astype(np.float32) / 255.0
        else:
            video_arr = load_video_frames(item["video_path"])  # (T, 96, 96)

        # Azami kare sınırlandırma (CTC kısıtını koru: T >= U)
        min_required_frames = len(token_indices) + 2
        effective_max = max(self.max_frames, min_required_frames)
        if video_arr.shape[0] > effective_max:
            video_arr = video_arr[: effective_max]

        # Veri artırma (Augmentation)
        video_aug = self._apply_augmentations(video_arr, target_len=len(token_indices))  # (T, 88, 88)

        # Standart Dudak Normalizasyonu: (x - mean) / std
        mean = np.mean(video_aug)
        std = np.std(video_aug) + 1e-6
        video_norm = (video_aug - mean) / std

        # PyTorch formatına dönüştür: (1, T, H, W)
        video_tensor = torch.from_numpy(video_norm).unsqueeze(0).float()

        return {
            "video": video_tensor,
            "target": target_tensor,
            "transcript": clean_text,
            "video_id": item.get("video_id", ""),
            "seg_id": item.get("seg_id", ""),
            "num_frames": video_tensor.size(1),
            "target_length": len(token_indices),
            "duration": item.get("duration", 0.0),
        }


class LipReadingWordDataset(Dataset):
    """
    Türkçe Dudak Okuma Kelime Düzeyi (Word-Level) Veri Kümesi.
    - align.json içindeki milisaniyelik kelime zaman damgalarını kullanarak
      96x96 gri dudak videolarından doğrudan Top-500 kelime dilimlerini (T_word, 96, 96) çıkarır.
    - Cümle düzeyindeki sessizlik ve boşluk (blank) patolojisini yapısal olarak yok eder.
    - Ana videoları (parent videos) RAM'de tekil olarak önbelleğe alarak bellek tasarrufu ve 30x+ hız sağlar.
    """

    def __init__(
        self,
        master_dir: Optional[Union[str, pathlib.Path]] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        is_train: bool = True,
        split: Optional[str] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        target_vocab: Optional[Set[str]] = None,
        crop_size: int = 88,
        context_frames: int = 1,
        context_jitter: int = 1,
        max_per_word: Optional[int] = 20,
        min_frames: int = 3,
        apply_horizontal_flip: bool = True,
        apply_time_masking: bool = True,
        cache_in_ram: bool = True,
    ):
        self.is_train = is_train
        self.crop_size = crop_size
        self.context_frames = context_frames
        self.context_jitter = context_jitter
        self.max_per_word = max_per_word
        self.min_frames = min_frames
        self.apply_horizontal_flip = apply_horizontal_flip and is_train
        self.apply_time_masking = apply_time_masking and is_train
        self.target_vocab = target_vocab or TOP_500_SET
        self.cache_in_ram = cache_in_ram

        self.word_samples: List[Dict[str, Any]] = []
        # Ana video dosya yolu -> uint8 video dizisi eşlemesi (RAM önbellek)
        self._parent_cache: Dict[str, np.ndarray] = {}

        if samples is not None:
            self.word_samples = samples
        elif master_dir is not None:
            self.word_samples = self._scan_word_samples(
                pathlib.Path(master_dir),
                split=split,
                split_map_path=split_map_path,
                target_vocab=self.target_vocab,
                context_frames=self.context_frames,
                min_frames=self.min_frames,
                max_per_word=self.max_per_word,
            )

        if self.cache_in_ram and len(self.word_samples) > 0:
            self._preload_parent_videos()

    @staticmethod
    def _scan_word_samples(
        master_path: pathlib.Path,
        split: Optional[str] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        target_vocab: Optional[Set[str]] = None,
        context_frames: int = 1,
        min_frames: int = 3,
        max_per_word: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Dizini tarayarak align.json'dan filtrelenmiş ve dengelenmiş geçerli kelime dilimlerini toplar."""
        collected = []
        if not master_path.exists():
            return collected

        allowed_videos = _load_allowed_videos(master_path, split, split_map_path)

        for align_file in sorted(master_path.glob("*/*/align.json")):
            seg_dir = align_file.parent
            vid = seg_dir.parent.name
            if allowed_videos is not None and vid not in allowed_videos:
                continue

            video_file = seg_dir / "mouth.mp4"
            if not video_file.exists():
                continue

            try:
                with open(align_file, "r", encoding="utf-8") as f:
                    align_data = json.load(f)
                words = align_data.get("words", [])
            except Exception:
                continue

            if not words:
                continue

            clip_offset = words[0].get("start", 0.0)

            for w_entry in words:
                raw_w = w_entry.get("word", "")
                norm_w = normalize_turkish_text(raw_w)
                if not norm_w:
                    continue

                if target_vocab is not None and norm_w not in target_vocab:
                    continue

                token_indices = text_to_indices(norm_w)
                L = len(token_indices)
                if L == 0:
                    continue

                rel_start = max(0.0, w_entry.get("start", 0.0) - clip_offset)
                rel_end = max(rel_start + 0.08, w_entry.get("end", 0.0) - clip_offset)

                start_f = max(0, int(rel_start * 25.0) - context_frames)
                end_f = int(rel_end * 25.0) + context_frames

                # CTC kısıtı: T >= L ve asgari kare sayısı
                T = end_f - start_f
                if T < L or T < min_frames:
                    continue

                collected.append({
                    "word": norm_w,
                    "tokens": token_indices,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "num_frames": T,
                    "target_length": L,
                    "video_path": str(video_file),
                    "video_id": vid,
                    "seg_id": seg_dir.name,
                })

        # Sınıf Dengeleme (Class Equalization / Frequency Capping):
        # 'bir', 'ben', 'var' gibi aşırı sık kelimelerin veri kümesini domine etmesini engeller.
        if max_per_word is not None and max_per_word > 0:
            by_word: Dict[str, List[Dict[str, Any]]] = {}
            for item in collected:
                by_word.setdefault(item["word"], []).append(item)

            balanced = []
            for word, items in sorted(by_word.items()):
                if len(items) <= max_per_word:
                    balanced.extend(items)
                else:
                    # Konuşmacı çeşitliliğini artırmak için video_id'ye göre sıralayıp eşit aralıkla örnekle
                    items.sort(key=lambda x: (x["video_id"], x["seg_id"], x["start_frame"]))
                    step = len(items) / float(max_per_word)
                    sampled = [items[int(i * step)] for i in range(max_per_word)]
                    balanced.extend(sampled)
            return balanced

        return collected

    def _preload_parent_videos(self) -> None:
        """Kelime dilimlerinin ait olduğu ana MP4 videolarını RAM'e bir kez alır."""
        import time
        t0 = time.time()
        unique_paths = sorted(list({item["video_path"] for item in self.word_samples}))
        print(f"📦 [{len(self.word_samples)} Kelime Dilimi için {len(unique_paths)} Ana Video] RAM'e önbelleğe alınıyor...")
        for p in unique_paths:
            try:
                raw_float = load_video_frames(p)
                raw_uint8 = (raw_float * 255.0).clip(0, 255).astype(np.uint8)
                self._parent_cache[p] = raw_uint8
            except Exception:
                pass
        elapsed = round(time.time() - t0, 2)
        total_mb = sum(arr.nbytes for arr in self._parent_cache.values()) / (1024 * 1024)
        print(f"✅ Kelime Önbelleği Hazır: {len(self.word_samples)} kelime dilimi, {total_mb:.1f} MB RAM ({elapsed}s).")

    def __len__(self) -> int:
        return len(self.word_samples)

    def _apply_augmentations(self, video: np.ndarray) -> np.ndarray:
        """(T, H, W) float32 -> (T, crop_size, crop_size) float32"""
        T, H, W = video.shape
        if H < self.crop_size or W < self.crop_size:
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            video = np.pad(video, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            T, H, W = video.shape

        if self.is_train:
            max_y = max(0, H - self.crop_size)
            max_x = max(0, W - self.crop_size)
            top = random.randint(0, max_y) if max_y > 0 else 0
            left = random.randint(0, max_x) if max_x > 0 else 0
        else:
            top = (H - self.crop_size) // 2
            left = (W - self.crop_size) // 2

        video = video[:, top : top + self.crop_size, left : left + self.crop_size]

        if self.apply_horizontal_flip and random.random() < 0.5:
            video = np.flip(video, axis=-1).copy()

        if self.apply_time_masking and random.random() < 0.3 and T > 6:
            mask_len = max(1, int(0.15 * T))
            mask_start = random.randint(0, T - mask_len)
            video[mask_start : mask_start + mask_len] = np.mean(video)

        return video

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.word_samples[idx]
        v_path = item["video_path"]
        s_f = item["start_frame"]
        e_f = item["end_frame"]
        L = item["target_length"]

        # Eğitim sırasında rastgele bağlam kayması (Context Jitter):
        # Modelin kelimeye giriş ve çıkıştaki ardışık artikülasyonlara (co-articulation) dayanıklılığını artırır.
        if self.is_train and self.context_jitter > 0:
            jit_s = random.randint(-self.context_jitter, self.context_jitter)
            jit_e = random.randint(-self.context_jitter, self.context_jitter)
            s_f = max(0, s_f + jit_s)
            e_f = max(s_f + L, e_f + jit_e)

        if v_path in self._parent_cache:
            parent_arr = self._parent_cache[v_path]
            T_total = parent_arr.shape[0]
            actual_end = min(e_f, T_total)
            actual_start = min(s_f, max(0, actual_end - 1))
            # CTC kısıtı: video süresi hedef uzunluğundan kısa olamaz
            if actual_end - actual_start < L:
                actual_start = max(0, actual_end - L)
            word_arr = parent_arr[actual_start:actual_end].astype(np.float32) / 255.0
        else:
            parent_float = load_video_frames(v_path)
            T_total = parent_float.shape[0]
            actual_end = min(e_f, T_total)
            actual_start = min(s_f, max(0, actual_end - 1))
            if actual_end - actual_start < L:
                actual_start = max(0, actual_end - L)
            word_arr = parent_float[actual_start:actual_end]

        if word_arr.shape[0] == 0:
            word_arr = np.zeros((item["num_frames"], 96, 96), dtype=np.float32)

        # Veri artırma ve normalizasyon
        video_aug = self._apply_augmentations(word_arr)
        mean = np.mean(video_aug)
        std = np.std(video_aug) + 1e-6
        video_norm = (video_aug - mean) / std

        video_tensor = torch.from_numpy(video_norm).unsqueeze(0).float()  # (1, T, H, W)
        target_tensor = torch.tensor(item["tokens"], dtype=torch.long)

        return {
            "video": video_tensor,
            "target": target_tensor,
            "transcript": item["word"],
            "word": item["word"],
            "video_id": item.get("video_id", ""),
            "seg_id": item.get("seg_id", ""),
            "num_frames": video_tensor.size(1),
            "target_length": len(item["tokens"]),
        }


class LipReadingPhraseDataset(Dataset):
    """
    Çoklu Kelimeli İfade Dilimleri (Phrase Slices) Veri Kümesi.
    align.json içindeki 2-4 ardışık kelimelik konuşma öbeklerini (25-80 kare) dilimleyerek
    izole tek kelimeler ile tam cümleler arasındaki köprüyü kurar.
    Modelin boşluklu kelime dizilerini ve kelimeler arası geçiş artikülasyonlarını öğrenmesini sağlar.
    """

    def __init__(
        self,
        master_dir: Optional[Union[str, pathlib.Path]] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        is_train: bool = True,
        split: Optional[str] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        crop_size: int = 88,
        min_words: int = 2,
        max_words: int = 4,
        max_phrase_frames: int = 80,
        apply_horizontal_flip: bool = True,
        apply_time_masking: bool = True,
        cache_in_ram: bool = True,
    ):
        self.is_train = is_train
        self.crop_size = crop_size
        self.min_words = min_words
        self.max_words = max_words
        self.max_phrase_frames = max_phrase_frames
        self.apply_horizontal_flip = apply_horizontal_flip and is_train
        self.apply_time_masking = apply_time_masking and is_train
        self.cache_in_ram = cache_in_ram

        self.phrase_samples: List[Dict[str, Any]] = []
        self._parent_cache: Dict[str, np.ndarray] = {}

        if samples is not None:
            self.phrase_samples = samples
        elif master_dir is not None:
            self.phrase_samples = self._scan_phrase_samples(
                pathlib.Path(master_dir),
                split=split,
                split_map_path=split_map_path,
                min_words=self.min_words,
                max_words=self.max_words,
                max_phrase_frames=self.max_phrase_frames,
            )

        if self.cache_in_ram and len(self.phrase_samples) > 0:
            self._preload_parent_videos()

    @staticmethod
    def _scan_phrase_samples(
        master_path: pathlib.Path,
        split: Optional[str] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        min_words: int = 2,
        max_words: int = 4,
        max_phrase_frames: int = 80,
    ) -> List[Dict[str, Any]]:
        collected = []
        if not master_path.exists():
            return collected

        allowed_videos = _load_allowed_videos(master_path, split, split_map_path)

        for align_file in sorted(master_path.glob("*/*/align.json")):
            seg_dir = align_file.parent
            vid = seg_dir.parent.name
            if allowed_videos is not None and vid not in allowed_videos:
                continue

            video_file = seg_dir / "mouth.mp4"
            if not video_file.exists():
                continue

            try:
                with open(align_file, "r", encoding="utf-8") as f:
                    align_data = json.load(f)
                words = align_data.get("words", [])
            except Exception:
                continue

            if len(words) < min_words:
                continue

            clip_offset = words[0].get("start", 0.0)
            n_w = len(words)

            i = 0
            while i < n_w:
                k = min(n_w - i, random.randint(min_words, max_words) if n_w - i >= min_words else n_w - i)
                if k < min_words:
                    break

                chunk = words[i : i + k]
                chunk_words = [normalize_turkish_text(w.get("word", "")) for w in chunk]
                chunk_words = [cw for cw in chunk_words if cw]
                if len(chunk_words) < min_words:
                    i += 1
                    continue

                phrase_text = " ".join(chunk_words)
                token_indices = text_to_indices(phrase_text)
                L = len(token_indices)

                rel_start = max(0.0, chunk[0].get("start", 0.0) - clip_offset)
                rel_end = max(rel_start + 0.1, chunk[-1].get("end", 0.0) - clip_offset)

                start_f = max(0, int(rel_start * 25.0))
                end_f = int(rel_end * 25.0)
                T = end_f - start_f

                if T >= L and T <= max_phrase_frames:
                    collected.append({
                        "phrase": phrase_text,
                        "tokens": token_indices,
                        "start_frame": start_f,
                        "end_frame": end_f,
                        "num_frames": T,
                        "target_length": L,
                        "video_path": str(video_file),
                        "video_id": vid,
                        "seg_id": seg_dir.name,
                    })
                    i += k
                else:
                    i += 1

        return collected

    def _preload_parent_videos(self) -> None:
        unique_paths = sorted(list({item["video_path"] for item in self.phrase_samples}))
        for p in unique_paths:
            try:
                raw_float = load_video_frames(p)
                raw_uint8 = (raw_float * 255.0).clip(0, 255).astype(np.uint8)
                self._parent_cache[p] = raw_uint8
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.phrase_samples)

    def _apply_augmentations(self, video: np.ndarray) -> np.ndarray:
        T, H, W = video.shape
        if H < self.crop_size or W < self.crop_size:
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            video = np.pad(video, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            T, H, W = video.shape

        if self.is_train:
            max_y = max(0, H - self.crop_size)
            max_x = max(0, W - self.crop_size)
            top = random.randint(0, max_y) if max_y > 0 else 0
            left = random.randint(0, max_x) if max_x > 0 else 0
        else:
            top = (H - self.crop_size) // 2
            left = (W - self.crop_size) // 2

        video = video[:, top : top + self.crop_size, left : left + self.crop_size]

        if self.apply_horizontal_flip and random.random() < 0.5:
            video = np.flip(video, axis=-1).copy()

        if self.apply_time_masking and random.random() < 0.3 and T > 8:
            mask_len = max(1, int(0.12 * T))
            mask_start = random.randint(0, T - mask_len)
            video[mask_start : mask_start + mask_len] = np.mean(video)

        return video

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.phrase_samples[idx]
        v_path = item["video_path"]
        s_f = item["start_frame"]
        e_f = item["end_frame"]
        L = item["target_length"]

        if v_path in self._parent_cache:
            parent_arr = self._parent_cache[v_path]
            T_total = parent_arr.shape[0]
            actual_end = min(e_f, T_total)
            actual_start = min(s_f, max(0, actual_end - 1))
            if actual_end - actual_start < L:
                actual_start = max(0, actual_end - L)
            phrase_arr = parent_arr[actual_start:actual_end].astype(np.float32) / 255.0
        else:
            parent_float = load_video_frames(v_path)
            T_total = parent_float.shape[0]
            actual_end = min(e_f, T_total)
            actual_start = min(s_f, max(0, actual_end - 1))
            if actual_end - actual_start < L:
                actual_start = max(0, actual_end - L)
            phrase_arr = parent_float[actual_start:actual_end]

        if phrase_arr.shape[0] == 0:
            phrase_arr = np.zeros((item["num_frames"], 96, 96), dtype=np.float32)

        video_aug = self._apply_augmentations(phrase_arr)
        mean = np.mean(video_aug)
        std = np.std(video_aug) + 1e-6
        video_norm = (video_aug - mean) / std

        video_tensor = torch.from_numpy(video_norm).unsqueeze(0).float()
        target_tensor = torch.tensor(item["tokens"], dtype=torch.long)

        return {
            "video": video_tensor,
            "target": target_tensor,
            "transcript": item["phrase"],
            "phrase": item["phrase"],
            "word": item["phrase"],
            "video_id": item.get("video_id", ""),
            "seg_id": item.get("seg_id", ""),
            "num_frames": video_tensor.size(1),
            "target_length": len(item["tokens"]),
        }


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Değişken uzunluklu video ve hedef dizileri dinamik olarak dolduran collate_fn.
    Çıktılar:
      - videos: (Batch, 1, T_max, 88, 88)
      - targets: (Batch, L_max)
      - input_lengths: (Batch,)
      - target_lengths: (Batch,)
      - transcripts: List[str]
      - video_ids: List[str]
      - seg_ids: List[str]
    """
    batch_size = len(batch)

    # Maksimum zamansal ve hedef boyutları bul
    max_frames = max(item["num_frames"] for item in batch)
    max_target_len = max(max(item["target_length"] for item in batch), 1)

    # Tensor tahsisleri
    _, _, h, w = batch[0]["video"].shape
    padded_videos = torch.zeros(batch_size, 1, max_frames, h, w, dtype=torch.float32)
    padded_targets = torch.full((batch_size, max_target_len), PAD_IDX, dtype=torch.long)

    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    transcripts = []
    video_ids = []
    seg_ids = []

    for i, item in enumerate(batch):
        t = item["num_frames"]
        l = item["target_length"]

        padded_videos[i, :, :t, :, :] = item["video"]
        if l > 0:
            padded_targets[i, :l] = item["target"]

        input_lengths[i] = t
        target_lengths[i] = l
        transcripts.append(item["transcript"])
        video_ids.append(item["video_id"])
        seg_ids.append(item["seg_id"])

    return {
        "videos": padded_videos,
        "targets": padded_targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "transcripts": transcripts,
        "video_ids": video_ids,
        "seg_ids": seg_ids,
    }


class SequenceBucketSampler(torch.utils.data.Sampler[List[int]]):
    """
    Değişken uzunluklu video sekanslarını uzunluklarına (duration veya num_frames)
    göre kümelere (bucket) ayırarak, benzer uzunluktaki klipleri bir arada gruplayan
    ve uzun sekanslar için daha küçük batch boyutu tahsis eden dinamik örnekleyici.

    Özellikler:
    1. Zero-padding israfını %70-%85 oranında azaltır.
    2. GPU VRAM taşmalarını (OOM) önler (ör. >8.0s klipler için batch=2-4).
    3. Epoch boyunca bucket'lar arası rastgele harmanlama (interleaving) yaparak
       gradyan yönünde uzunluk yanlılığını engeller.
    """

    def __init__(
        self,
        dataset: Any,
        buckets: Optional[List[Tuple[float, float, int]]] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Varsayılan bucket'lar: (min_duration_sec, max_duration_sec, batch_size)
        # 0.0 - 3.5s : batch_size = 8 (Kısa klipler: kelime ve kısa ifadeler)
        # 3.5 - 8.0s : batch_size = 6 (Orta klipler: standart cümleler)
        # > 8.0s     : batch_size = 2 (Uzun klipler: karmaşık cümleler, VRAM güvenliği)
        self.buckets = buckets or [
            (0.0, 3.5, 8),
            (3.5, 8.0, 6),
            (8.0, 100.0, 2),
        ]

        # Örnek sürelerini topla
        self.sample_durations: List[float] = []
        if hasattr(dataset, "samples"):
            for s in dataset.samples:
                dur = float(s.get("duration", 0.0))
                if dur <= 0.0 and "num_frames" in s:
                    dur = s["num_frames"] / 25.0
                self.sample_durations.append(dur)
        else:
            self.sample_durations = [0.0] * len(dataset)

        # İndeksleri bucket'lara dağıt
        self.bucket_indices: List[List[int]] = [[] for _ in self.buckets]
        for idx, dur in enumerate(self.sample_durations):
            placed = False
            for b_idx, (b_min, b_max, _) in enumerate(self.buckets):
                if b_min <= dur < b_max:
                    self.bucket_indices[b_idx].append(idx)
                    placed = True
                    break
            if not placed:
                self.bucket_indices[-1].append(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batches: List[List[int]] = []

        for b_idx, (_, _, batch_size) in enumerate(self.buckets):
            indices = list(self.bucket_indices[b_idx])
            if not indices:
                continue
            if self.shuffle:
                rng.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                if self.drop_last and len(batch) < batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for b_idx, (_, _, batch_size) in enumerate(self.buckets):
            n = len(self.bucket_indices[b_idx])
            if n == 0:
                continue
            if self.drop_last:
                total_batches += n // batch_size
            else:
                total_batches += (n + batch_size - 1) // batch_size
        return total_batches
