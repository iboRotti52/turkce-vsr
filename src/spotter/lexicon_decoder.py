"""
src/spotter/lexicon_decoder.py — 500 Kelimelik Türkçe Sözlük Kısıtlı CTC Prefix Beam Search Çözücü
Sürekli konuşma videolarında görsel fonetik belirsizliği (visem/homofen) sözlük kısıtlamasıyla aşarak
yalnızca geçerli Türkçe kelimeleri, tam başlangıç/bitiş zaman damgalarıyla çözer.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.spotter.keyword_spotter import DetectedKeyword
from src.vocab.top500_words import TOP_500_SET, TOP_500_WORDS, WORD_TO_RANK, get_word_viseme_signature
from src.vocab.turkish_lm import TurkishLanguageModel
from src.vocab.turkish_vocab import (
    BLANK_IDX,
    IDX_TO_CHAR,
    PAD_IDX,
    SPACE_IDX,
    TURKISH_LETTERS,
    VISEME_MAP,
    VOCAB_SIZE,
    normalize_turkish_text,
)


def _logsumexp(a: float, b: float) -> float:
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


class TrieNode:
    def __init__(self, char: str = ""):
        self.char = char
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word = False
        self.word = ""


class LexiconBeamSearchDecoder:
    """
    Hedef 500 kelimelik Trie ağacı üzerinde çalışan, visem benzerliği ve
    Türkçe Unigram/Bigram Dil Modeli (LM) geçiş önsel olasılıklarıyla
    güçlendirilmiş CTC Prefix Beam Search çözücüsü.

    Puanlama Formülü:
      Score = log P_CTC + alpha * log P_LM(W) + beta * WordCount
    """

    def __init__(
        self,
        target_vocab: Optional[Set[str]] = None,
        beam_size: int = 50,
        fps: float = 25.0,
        blank_penalty: float = 0.2,
        viseme_tolerance: bool = True,
        lm_alpha: float = 0.4,
        lm_beta: float = 1.0,
        repeat_penalty: float = 3.0,
        min_dur_factor: float = 1.0,
        use_lm: bool = True,
        lm: Optional[TurkishLanguageModel] = None,
    ):
        self.target_vocab = target_vocab or TOP_500_SET
        self.beam_size = beam_size
        self.fps = fps
        self.blank_penalty = blank_penalty
        self._viseme_tolerance = viseme_tolerance
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.repeat_penalty = repeat_penalty
        self.min_dur_factor = min_dur_factor
        self.use_lm = use_lm
        self.lm = lm or (TurkishLanguageModel() if use_lm else None)

        # Trie inşa et
        self.root = TrieNode()
        self._build_trie()
        self._precompute_viseme_table()

    @property
    def viseme_tolerance(self) -> bool:
        return self._viseme_tolerance

    @viseme_tolerance.setter
    def viseme_tolerance(self, value: bool) -> None:
        self._viseme_tolerance = value
        self._precompute_viseme_table()

    def _build_trie(self) -> None:
        words = [w for w in TOP_500_WORDS if w in self.target_vocab and len(w) >= 2]
        for w in words:
            curr = self.root
            for ch in w:
                if ch not in curr.children:
                    curr.children[ch] = TrieNode(ch)
                curr = curr.children[ch]
            curr.is_word = True
            curr.word = w

    def _get_viseme_distance(self, expected_ch: str, emitted_ch: str) -> float:
        """İki karakter arasındaki görsel visem mesafesi cezası (log-prob alanı)."""
        if expected_ch == emitted_ch:
            return 0.0
        v1 = VISEME_MAP.get(expected_ch, -1)
        v2 = VISEME_MAP.get(emitted_ch, -2)
        if v1 == v2 and v1 != -1:
            return 0.35  # Aynı visem grubu (örn: b ve p)
        if {v1, v2} == {0, 1}:
            return 0.55  # Çift dudaksı ve Diş-dudaksı yakınlığı (örn: b ve v)
        if {v1, v2} == {2, 3}:
            return 0.75  # Dişeti ve damaksı yakınlığı (örn: d ve c)
        return 2.50  # Görsel olarak tamamen farklı artikülasyon

    def _precompute_viseme_table(self) -> None:
        self._viseme_match_table: Dict[str, List[Tuple[str, float]]] = {}
        for ch in TURKISH_LETTERS:
            if not self._viseme_tolerance:
                self._viseme_match_table[ch] = [(ch, 0.0)]
                continue
            matches = []
            for exp in TURKISH_LETTERS:
                d = self._get_viseme_distance(exp, ch)
                if d <= 1.0:
                    matches.append((exp, d))
            self._viseme_match_table[ch] = matches

    def decode(
        self,
        logits: torch.Tensor,
        beam_size: Optional[int] = None,
        blank_penalty: Optional[float] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        min_dur_factor: Optional[float] = None,
    ) -> Tuple[str, List[DetectedKeyword]]:
        """
        Modelin (T, Vocab) veya (1, T, Vocab) logits tensörünü çözer.
        Dönen değer: (en_iyi_metin, yakalanan_kelimeler_listesi)
        """
        if logits.dim() == 3:
            logits = logits[0]

        bs = beam_size or self.beam_size
        bp = self.blank_penalty if blank_penalty is None else blank_penalty
        alpha = self.lm_alpha if lm_alpha is None else lm_alpha
        beta = self.lm_beta if lm_beta is None else lm_beta
        rep_pen = self.repeat_penalty if repeat_penalty is None else repeat_penalty
        mdf = self.min_dur_factor if min_dur_factor is None else min_dur_factor

        logits_adj = logits.clone()
        if bp > 0.0:
            logits_adj[:, BLANK_IDX] -= bp

        log_probs = logits_adj.log_softmax(-1).cpu()
        T = log_probs.size(0)

        # Durum: (words_tuple, current_node, word_start_frame, last_char_frame, last_emitted_char) -> (p_b, p_nb)
        # words_tuple: ((word, start_frame, end_frame), ...)
        init_state = ((), self.root, 0, 0, "")
        beam = {init_state: (0.0, -float("inf"))}

        for t in range(T):
            lp = log_probs[t]
            new_beam = defaultdict(lambda: (-float("inf"), -float("inf")))

            # En iyi adayları seç
            sorted_beam = sorted(
                beam.items(),
                key=lambda item: _logsumexp(item[1][0], item[1][1]),
                reverse=True,
            )[:bs]

            top_vals, top_idxs = torch.topk(lp, k=8)
            best_eff_by_char: Dict[str, float] = {}
            for v, idx in zip(top_vals, top_idxs):
                idx_i = idx.item()
                v_i = v.item()
                if idx_i in (BLANK_IDX, PAD_IDX, SPACE_IDX) or v_i < -4.0:
                    continue
                em_ch = IDX_TO_CHAR.get(idx_i, "")
                if not em_ch:
                    continue
                for exp_ch, d in self._viseme_match_table.get(em_ch, [(em_ch, 0.0)]):
                    eff = v_i - d
                    if eff > best_eff_by_char.get(exp_ch, -float("inf")):
                        best_eff_by_char[exp_ch] = eff

            lp_blank = lp[BLANK_IDX].item()
            lp_space = lp[SPACE_IDX].item()

            for (words, node, w_start, last_char_f, last_ch), (p_b, p_nb) in sorted_beam:
                p_tot = _logsumexp(p_b, p_nb)

                # 1a. Blank emisyonu: Aynı düğümde kal (kelime içi blank veya root'ta sessizlik)
                cur_b, cur_nb = new_beam[(words, node, w_start, last_char_f, "")]
                new_beam[(words, node, w_start, last_char_f, "")] = (
                    _logsumexp(cur_b, p_tot + lp_blank),
                    cur_nb,
                )

                # 1b. Blank emisyonu ile kelime tamamlama (inter-word blank / sözcük arası sessizlik)
                if node.is_word and node != self.root:
                    completed_word = node.word
                    dur_frames = max(1, last_char_f - w_start + 1)
                    min_dur = max(2, int(len(completed_word) * mdf))
                    max_dur = max(12, len(completed_word) * 8)
                    if min_dur <= dur_frames <= max_dur:
                        new_words = words + ((completed_word, w_start, last_char_f),)
                        prev_w = words[-1][0] if words else None
                        lm_score = 0.0
                        if self.use_lm and self.lm is not None:
                            lm_score = alpha * self.lm.score_transition(prev_w, completed_word)
                        lm_score += beta
                        if prev_w == completed_word:
                            lm_score -= rep_pen
                        cur_b_root, cur_nb_root = new_beam[(new_words, self.root, t, t, "")]
                        new_beam[(new_words, self.root, t, t, "")] = (
                            _logsumexp(cur_b_root, p_tot + lp_blank + lm_score),
                            cur_nb_root,
                        )

                # 2. Boşluk (Space) emisyonu: Kelime tamamlanmışsa ve akustik kanıt varsa yeni kelimeye geç
                if lp_space > -4.5 and node.is_word and node != self.root:
                    completed_word = node.word
                    dur_frames = max(1, last_char_f - w_start + 1)
                    min_dur = max(2, int(len(completed_word) * mdf))
                    max_dur = max(12, len(completed_word) * 8)
                    if min_dur <= dur_frames <= max_dur:
                        new_words = words + ((completed_word, w_start, last_char_f),)
                        prev_w = words[-1][0] if words else None
                        lm_score = 0.0
                        if self.use_lm and self.lm is not None:
                            lm_score = alpha * self.lm.score_transition(prev_w, completed_word)
                        lm_score += beta
                        if prev_w == completed_word:
                            lm_score -= rep_pen
                        cur_b_sp, cur_nb_sp = new_beam[(new_words, self.root, t + 1, t + 1, " ")]
                        new_beam[(new_words, self.root, t + 1, t + 1, " ")] = (
                            cur_b_sp,
                            _logsumexp(cur_nb_sp, p_tot + lp_space + lm_score),
                        )

                # 3. Mevcut kelimeyi uzatan harf emisyonu (node.children)
                for exp_ch in node.children:
                    if exp_ch in best_eff_by_char:
                        next_node = node.children[exp_ch]
                        eff_lp = best_eff_by_char[exp_ch]
                        new_w_start = t if node == self.root else w_start
                        st = (words, next_node, new_w_start, t, exp_ch)
                        cur_b_ch, cur_nb_ch = new_beam[st]
                        if exp_ch == last_ch:
                            new_beam[st] = (cur_b_ch, _logsumexp(cur_nb_ch, p_b + eff_lp))
                        else:
                            new_beam[st] = (cur_b_ch, _logsumexp(cur_nb_ch, p_tot + eff_lp))

                # 4. Doğrudan kelimeler arası geçiş / koartikülasyon (node.is_word -> root.children)
                if node.is_word and node != self.root:
                    completed_word = node.word
                    dur_frames = max(1, last_char_f - w_start + 1)
                    min_dur = max(2, int(len(completed_word) * mdf))
                    max_dur = max(12, len(completed_word) * 8)
                    if min_dur <= dur_frames <= max_dur:
                        new_words = words + ((completed_word, w_start, last_char_f),)
                        prev_w = words[-1][0] if words else None
                        lm_score = 0.0
                        if self.use_lm and self.lm is not None:
                            lm_score = alpha * self.lm.score_transition(prev_w, completed_word)
                        lm_score += beta
                        if prev_w == completed_word:
                            lm_score -= rep_pen
                        for exp_ch in self.root.children:
                            if exp_ch in best_eff_by_char:
                                next_node = self.root.children[exp_ch]
                                eff_lp = best_eff_by_char[exp_ch]
                                st = (new_words, next_node, t, t, exp_ch)
                                cur_b_co, cur_nb_co = new_beam[st]
                                new_beam[st] = (
                                    cur_b_co,
                                    _logsumexp(cur_nb_co, p_tot + eff_lp + lm_score),
                                )

            beam = new_beam

        # En yüksek olasılıklı hipotezi seç
        scored = []
        for (words, node, w_s, last_char_f, last_ch), (p_b, p_nb) in beam.items():
            tot = _logsumexp(p_b, p_nb)
            final_w = words
            if node.is_word and node != self.root:
                dur_frames = max(1, last_char_f - w_s + 1)
                min_dur = max(2, int(len(node.word) * mdf))
                max_dur = max(12, len(node.word) * 8)
                if min_dur <= dur_frames <= max_dur:
                    prev_w = words[-1][0] if words else None
                    lm_score = 0.0
                    if self.use_lm and self.lm is not None:
                        lm_score = alpha * self.lm.score_transition(prev_w, node.word)
                    lm_score += beta
                    if prev_w == node.word:
                        lm_score -= rep_pen
                    tot += lm_score
                    final_w = words + ((node.word, w_s, last_char_f),)
            scored.append((tot, final_w))

        scored.sort(reverse=True)
        best_words = scored[0][1] if scored else ()

        detected_list: List[DetectedKeyword] = []
        for word, s_f, e_f in best_words:
            dur_f = max(1, e_f - s_f)
            s_sec = round(s_f / self.fps, 2)
            e_sec = round(e_f / self.fps, 2)
            rank = WORD_TO_RANK.get(word, -1)
            conf = min(0.99, max(0.40, round(1.0 - (1.0 / (dur_f + 1)), 3)))
            detected_list.append(
                DetectedKeyword(
                    word=word,
                    start_frame=s_f,
                    end_frame=e_f,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    confidence=conf,
                    rank_in_500=rank,
                    is_viseme_approx=True,
                    match_type="lexicon_beam",
                )
            )

        full_text = " ".join([d.word for d in detected_list])
        return full_text, detected_list
