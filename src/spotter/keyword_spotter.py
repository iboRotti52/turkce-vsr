"""
src/spotter/keyword_spotter.py — Sürekli Konuşmada 500 Kelime Avcısı (Keyword Spotter)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F

from src.vocab.turkish_vocab import (
    BLANK_IDX,
    PAD_IDX,
    UNK_IDX,
    SPACE_IDX,
    IDX_TO_CHAR,
    normalize_turkish_text,
    text_to_indices,
    to_viseme_sequence,
)
from src.vocab.top500_words import (
    TOP_500_SET,
    TOP_500_WORDS,
    WORD_TO_RANK,
    get_word_viseme_signature,
    is_top500_word,
)


@dataclass
class DetectedKeyword:
    word: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    confidence: float
    rank_in_500: int
    is_viseme_approx: bool = False
    match_type: str = "exact"  # "exact", "substring", "homophene"


class KeywordSpotter:
    """
    Sürekli VSR CTC çıktısı üzerinde hedef 500 Türkçe kelimeyi tespit eder.
    FPS varsayılan olarak 25'tir (kare -> saniye dönüşümü).
    Hem boşluklu ayrık konuşmayı hem de boşluksuz sürekli (silent continuous) konuşmayı destekler.
    """

    def __init__(
        self,
        target_vocab: Optional[Set[str]] = None,
        fps: float = 25.0,
        viseme_tolerance: bool = True,
        min_confidence: float = 0.25,
        blank_penalty: float = 0.0,
        allow_substring: bool = True,
    ):
        self.target_vocab = target_vocab or TOP_500_SET
        self.fps = fps
        self.viseme_tolerance = viseme_tolerance
        self.min_confidence = min_confidence
        self.blank_penalty = blank_penalty
        self.allow_substring = allow_substring

        # Hızlı viseme ters haritası: signature -> set(words)
        self.viseme_to_target: Dict[str, Set[str]] = {}
        for w in self.target_vocab:
            sig = get_word_viseme_signature(w)
            if sig:
                self.viseme_to_target.setdefault(sig, set()).add(w)

        # Uzunluklarına göre azalan sırada sıralanmış 500 kelime listesi (uzun kelimeler önce eşleşsin)
        self._words_by_length = sorted(
            [w for w in self.target_vocab if len(w) >= 2],
            key=lambda x: (len(x), -WORD_TO_RANK.get(x, 9999)),
            reverse=True,
        )

        # Kayan pencere CTC sözlük hizalama tensörlerini hazırla
        self._candidate_words = None
        self._batch_targets = None
        self._target_lengths = None
        self._ctc_loss_fn = None
        self._init_batched_targets()

    def _init_batched_targets(self) -> None:
        """Hedef sözlükteki tüm kelimeleri tek seferde CTC ile puanlamak için tensör önbelleği hazırlar."""
        cand_list = [w for w in TOP_500_WORDS if w in self.target_vocab]
        if not cand_list:
            cand_list = sorted(list(self.target_vocab))

        tokens_list = [text_to_indices(w) for w in cand_list]
        lens = [len(t) for t in tokens_list]
        max_l = max(max(lens), 1)

        b_targets = torch.zeros((len(cand_list), max_l), dtype=torch.long)
        for i, t in enumerate(tokens_list):
            if t:
                b_targets[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        self._candidate_words = cand_list
        self._batch_targets = b_targets
        self._target_lengths = torch.tensor(lens, dtype=torch.long)
        self._ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, reduction="none", zero_infinity=True)

    def spot_from_logits(
        self,
        logits: torch.Tensor,
        blank_penalty: Optional[float] = None,
        min_confidence: Optional[float] = None,
        viseme_tolerance: Optional[bool] = None,
        allow_substring: Optional[bool] = None,
    ) -> List[DetectedKeyword]:
        """
        Modelin ürettiği (T, Vocab) veya (1, T, Vocab) logits tensöründen
        hedef 500 kelimeleri zaman damgalarıyla çıkarır.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # (1, Vocab)
        elif logits.dim() == 3:
            if logits.size(0) == 1:
                logits = logits.squeeze(0)  # (T, Vocab)
            else:
                logits = logits[0]

        bp = self.blank_penalty if blank_penalty is None else blank_penalty
        min_conf = self.min_confidence if min_confidence is None else min_confidence
        vis_tol = self.viseme_tolerance if viseme_tolerance is None else viseme_tolerance
        allow_sub = self.allow_substring if allow_substring is None else allow_substring
        if bp > 0.0:
            logits = logits.clone()
            logits[:, BLANK_IDX] -= bp

        probs = torch.softmax(logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1).tolist()
        pred_confs = torch.max(probs, dim=-1).values.tolist()

        T = len(pred_indices)
        tokens = []

        # 1. CTC sıkıştırma: ardışık aynı indeksleri birleştir ama süre bilgisini tut
        t = 0
        while t < T:
            idx = pred_indices[t]
            start_t = t
            conf_sum = pred_confs[t]
            count = 1
            while t + 1 < T and pred_indices[t + 1] == idx:
                t += 1
                conf_sum += pred_confs[t]
                count += 1

            avg_conf = conf_sum / count
            # BLANK, PAD ve UNK özel tokenlarını kelime karakterlerine karıştırma
            if idx not in (BLANK_IDX, PAD_IDX, UNK_IDX):
                char = IDX_TO_CHAR.get(idx, "")
                if char:
                    tokens.append({
                        "char": char,
                        "start": start_t,
                        "end": t,
                        "conf": avg_conf,
                    })
            t += 1

        detections: List[DetectedKeyword] = []
        covered_spans: Set[Tuple[int, int]] = set()

        # 2. Boşluklara göre ayrık kelimeleri grupla ve eşleştir
        words_found = []
        current_word_chars = []
        current_start = 0
        current_end = 0
        current_confs = []

        for item in tokens:
            if item["char"] == " ":
                if current_word_chars:
                    raw_word = "".join(current_word_chars)
                    words_found.append({
                        "word": raw_word,
                        "start": current_start,
                        "end": current_end,
                        "conf": sum(current_confs) / max(1, len(current_confs)),
                    })
                    current_word_chars = []
                    current_confs = []
            else:
                if not current_word_chars:
                    current_start = item["start"]
                current_word_chars.append(item["char"])
                current_end = item["end"]
                current_confs.append(item["conf"])

        if current_word_chars:
            raw_word = "".join(current_word_chars)
            words_found.append({
                "word": raw_word,
                "start": current_start,
                "end": current_end,
                "conf": sum(current_confs) / max(1, len(current_confs)),
            })

        for w_info in words_found:
            w = normalize_turkish_text(w_info["word"])
            if not w or w_info["conf"] < min_conf:
                continue

            start_f = w_info["start"]
            end_f = w_info["end"]
            start_s = round(start_f / self.fps, 2)
            end_s = round(end_f / self.fps, 2)
            conf = round(w_info["conf"], 3)

            # Doğrudan tam eşleşme
            if w in self.target_vocab:
                rank = WORD_TO_RANK.get(w, -1)
                detections.append(DetectedKeyword(
                    word=w,
                    start_frame=start_f,
                    end_frame=end_f,
                    start_sec=start_s,
                    end_sec=end_s,
                    confidence=conf,
                    rank_in_500=rank,
                    is_viseme_approx=False,
                    match_type="exact",
                ))
                covered_spans.add((start_f, end_f))
            elif vis_tol:
                # Homophene (Viseme) toleransı
                sig = get_word_viseme_signature(w)
                if sig:
                    candidates = self.viseme_to_target.get(sig, set())
                    if candidates:
                        best_cand = min(candidates, key=lambda c: WORD_TO_RANK.get(c, 9999))
                        penalized_conf = round(conf * 0.9, 3)
                        if penalized_conf >= min_conf:
                            detections.append(DetectedKeyword(
                                word=best_cand,
                                start_frame=start_f,
                                end_frame=end_f,
                                start_sec=start_s,
                                end_sec=end_s,
                                confidence=penalized_conf,
                                rank_in_500=WORD_TO_RANK.get(best_cand, -1),
                                is_viseme_approx=True,
                                match_type="homophene",
                            ))
                            covered_spans.add((start_f, end_f))

        # 3. Sürekli Akış Alt-Dize (Continuous Substring) Eşleştirmesi
        # Boşluksuz veya ek almış sürekli konuşmada kelimeleri yakalar
        if allow_sub:
            non_space_tokens = [t for t in tokens if t["char"] != " "]
            if non_space_tokens:
                char_stream = "".join(t["char"] for t in non_space_tokens)
                n_stream = len(char_stream)

                for target_w in self._words_by_length:
                    # 2 harfli kelimeleri sadece izole veya belirgin durumlarda kabul et
                    if len(target_w) < 3 and len(char_stream) > 4:
                        continue

                    w_len = len(target_w)
                    find_idx = 0
                    while find_idx <= n_stream - w_len:
                        pos = char_stream.find(target_w, find_idx)
                        if pos == -1:
                            break

                        matched_tokens = non_space_tokens[pos : pos + w_len]
                        start_f = matched_tokens[0]["start"]
                        end_f = matched_tokens[-1]["end"]

                        # Zaten tam eşleşmeyle kapsanmış bir aralık değilse ekle
                        is_already_covered = any(
                            (cs <= start_f and end_f <= ce) for cs, ce in covered_spans
                        )
                        if not is_already_covered:
                            avg_conf = sum(t["conf"] for t in matched_tokens) / w_len
                            if avg_conf >= min_conf:
                                start_s = round(start_f / self.fps, 2)
                                end_s = round(end_f / self.fps, 2)
                                detections.append(DetectedKeyword(
                                    word=target_w,
                                    start_frame=start_f,
                                    end_frame=end_f,
                                    start_sec=start_s,
                                    end_sec=end_s,
                                    confidence=round(avg_conf, 3),
                                    rank_in_500=WORD_TO_RANK.get(target_w, -1),
                                    is_viseme_approx=False,
                                    match_type="substring",
                                ))
                                covered_spans.add((start_f, end_f))

                        find_idx = pos + 1

        # Zaman damgasına göre sırala
        detections.sort(key=lambda d: (d.start_frame, d.end_frame))
        return detections

    def spot_batch(
        self, logits: torch.Tensor, blank_penalty: Optional[float] = None
    ) -> List[List[DetectedKeyword]]:
        """
        Batch formatındaki logits tensöründen (B, T, Vocab) her örnek için
        tespit edilen anahtar kelimeleri döner.
        """
        bp = self.blank_penalty if blank_penalty is None else blank_penalty
        if logits.dim() == 1:
            return [self.spot_from_logits(logits.unsqueeze(0), blank_penalty=bp)]
        elif logits.dim() == 2:
            return [self.spot_from_logits(logits, blank_penalty=bp)]
        elif logits.dim() == 3:
            return [
                self.spot_from_logits(logits[b], blank_penalty=bp)
                for b in range(logits.size(0))
            ]
        else:
            raise ValueError(f"Beklenen logits boyutu 1, 2 veya 3, alınan: {logits.dim()}")

    def spot_from_text(self, text: str) -> List[str]:
        """
        Düz metin içindeki hedef 500 kelimeleri sıralı liste olarak döner.
        Hem boşluklu cümleleri hem de boşluksuz sürekli metinleri ve homofenleri yakalar.
        """
        norm = normalize_turkish_text(text)
        detected_words: List[str] = []
        covered_indices: Set[int] = set()

        # 1. Boşlukla ayrılmış kelimeleri kontrol et
        words = norm.split()
        for w in words:
            if w in self.target_vocab:
                detected_words.append(w)
            elif self.viseme_tolerance:
                sig = get_word_viseme_signature(w)
                if sig and sig in self.viseme_to_target:
                    candidates = self.viseme_to_target[sig]
                    best_cand = min(candidates, key=lambda c: WORD_TO_RANK.get(c, 9999))
                    detected_words.append(best_cand)

        # 2. Boşluksuz veya ek almış metinler için alt dize taraması
        no_spaces = norm.replace(" ", "")
        if len(detected_words) == 0 and len(no_spaces) >= 2:
            for target_w in self._words_by_length:
                if len(target_w) < 2:
                    continue
                pos = no_spaces.find(target_w)
                if pos != -1:
                    span_indices = set(range(pos, pos + len(target_w)))
                    if not (span_indices & covered_indices):
                        detected_words.append(target_w)
                        covered_indices.update(span_indices)

        return detected_words

    def spot_sliding_window_ctc(
        self,
        model: torch.nn.Module,
        video: torch.Tensor,
        window_sizes: Tuple[int, ...] = (10, 14, 18),
        stride: int = 3,
        min_score: float = -3.2,
        prior_weight: float = 0.6,
        topk_per_window: int = 3,
        iou_threshold: float = 0.4,
    ) -> List[DetectedKeyword]:
        """
        Sürekli konuşma videosu (1, 1, T, H, W) veya (1, T, D) özellik tensörü üzerinde
        kayan pencere CTC Sözlük Hizalama (Sliding Window CTC Lexicon Alignment) uygular.
        - 3D-ResNet frontend tüm video için 1 KEZ çalışır (aşırı hızlı).
        - Her pencere [t_s : t_e] BiGRU'ya verilerek t=0 başlangıç dinamikleri korunur.
        - Arka plan harf sıklığı (prior) çıkarılarak 'bir/ben' yanlılığı nötrlenir.
        - Non-Maximum Suppression (NMS) ile çakışmalar elenir.
        """
        device = next(model.parameters()).device
        model.eval()

        if self._candidate_words is None:
            self._init_batched_targets()

        batch_targets = self._batch_targets.to(device)
        target_lengths = self._target_lengths.to(device)

        with torch.no_grad():
            if video.dim() == 5:
                feats = model.frontend(video.to(device))  # (1, T, d_model)
                T = feats.size(1)
            elif video.dim() == 3:
                feats = video.to(device)
                T = feats.size(1)
            elif video.dim() == 2:
                feats = None
                logits_all = video.to(device)
                T = logits_all.size(0)
            else:
                raise ValueError(f"Beklenen video boyutu 2, 3 veya 5, alınan: {video.dim()}")

            # Video genelindeki ortalama arka plan harf dağılımı (prior)
            if feats is not None:
                full_logits = model.forward_from_features(feats)[0]  # (T, Vocab)
            else:
                full_logits = logits_all

            V_dim = full_logits.size(-1)
            bg_prior = full_logits.log_softmax(-1).mean(dim=0)  # (Vocab,)

            raw_detections = []

            for W in window_sizes:
                if W > T:
                    continue
                valid_mask = (target_lengths <= (W - 1)) & (target_lengths >= max(2, W // 4))
                if not valid_mask.any():
                    continue

                sub_targets = batch_targets[valid_mask]
                sub_lengths = target_lengths[valid_mask]
                valid_indices = torch.where(valid_mask)[0]
                n_valid = len(valid_indices)

                in_lens = torch.full((n_valid,), W, dtype=torch.long, device=device)

                for start_f in range(0, T - W + 1, stride):
                    end_f = start_f + W

                    if feats is not None:
                        sub_feats = feats[:, start_f:end_f, :]
                        sub_logits = model.forward_from_features(sub_feats)[0]  # (W, Vocab)
                    else:
                        sub_logits = logits_all[start_f:end_f]

                    # Arka plan prior çıkarımı ile normalizasyon
                    adj_lp = sub_logits.log_softmax(-1) - prior_weight * bg_prior.unsqueeze(0)
                    adj_lp = adj_lp.log_softmax(-1)

                    log_probs_exp = adj_lp.unsqueeze(1).expand(W, n_valid, V_dim)

                    losses = self._ctc_loss_fn(log_probs_exp, sub_targets, in_lens, sub_lengths)
                    scores = -losses / sub_lengths.float()

                    k = min(topk_per_window, n_valid)
                    top_scores, top_idx = torch.topk(scores, k=k)

                    for si in range(k):
                        sc = top_scores[si].item()
                        if sc >= min_score:
                            cand_idx = valid_indices[top_idx[si]].item()
                            w = self._candidate_words[cand_idx]
                            conf = round(torch.exp(torch.tensor(sc)).item(), 3)
                            raw_detections.append({
                                "word": w,
                                "start_frame": start_f,
                                "end_frame": end_f,
                                "score": sc,
                                "conf": conf,
                            })

        # Non-Maximum Suppression (NMS)
        raw_detections.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for det in raw_detections:
            s1, e1 = det["start_frame"], det["end_frame"]
            w1 = det["word"]
            overlap = False
            for kept in keep:
                s2, e2 = kept["start_frame"], kept["end_frame"]
                w2 = kept["word"]
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = max(e1, e2) - min(s1, s2)
                iou = inter / float(union) if union > 0 else 0.0
                # Aynı kelime ise en ufak zaman örtüşmesinde, farklı kelimeler ise iou > eşikte baskıla
                if (w1 == w2 and inter > 0) or (iou > iou_threshold):
                    overlap = True
                    break
            if not overlap:
                keep.append(det)

        keep.sort(key=lambda d: (d["start_frame"], d["end_frame"]))

        final_detections: List[DetectedKeyword] = []
        for d in keep:
            w = d["word"]
            s_f = d["start_frame"]
            e_f = d["end_frame"]
            final_detections.append(DetectedKeyword(
                word=w,
                start_frame=s_f,
                end_frame=e_f,
                start_sec=round(s_f / self.fps, 2),
                end_sec=round(e_f / self.fps, 2),
                confidence=d["conf"],
                rank_in_500=WORD_TO_RANK.get(w, -1),
                is_viseme_approx=False,
                match_type="ctc_alignment",
            ))

        return final_detections
