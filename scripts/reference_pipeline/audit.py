"""audit.py — VERİ-FİZİBİLİTE DENETİMİ (ucuz, GPU yok, eğitim yok).

state.db (qc_pass=1 segmentler) üzerinden, "YouTube verimiz fizibıl mi ve hangi
hedef gerçekçi?" sorusunu veri tarafından cevaplayan bir rapor üretir:

  - Ölçek:        toplam saat, segment, konuşmacı (≈video_id), süre dağılımı
  - Kapalı-küme:  kelime-frekans -> en sık N kelime; >=K örneği olan kelime sayısı
  - Dil kapsamı:  TR alfabe kapsamı; vizem grupları + homofen yoğunluğu (görsel
                  ayırt edilemezlik) -> kapalı/açık küme kararına ışık tutar

Çıktı: stdout özet + data/feasibility_report.md

Kullanım:
    python audit.py            # N=50, K=20 varsayılan
    python audit.py 100 30     # top-100 kelime, eşik 30 örnek
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

from config import BASE, DB_PATH, TR_ALPHABET

# Türkçe yaklaşık vizem (görsel ağız şekli) sınıfları. Aynı sınıftaki sesler
# dudaktan ayırt edilemez (homofen). Kaba ama fizibilite için yeterli sinyal.
VISEME = {}
for ch in "bpm":   VISEME[ch] = "P"   # çift dudak
for ch in "fv":    VISEME[ch] = "F"   # diş-dudak
for ch in "oöuü":  VISEME[ch] = "O"   # yuvarlak ünlü
for ch in "aeıi":  VISEME[ch] = "A"   # düz ünlü
for ch in "tdnszlr": VISEME[ch] = "T" # diş/diş-eti
for ch in "şçcj":  VISEME[ch] = "S"   # damak-diş
for ch in "kgğh":  VISEME[ch] = "K"   # art damak/gırtlak (ğ çoğu yerde sessiz)
VISEME["y"] = "Y"


def viseme_sig(word: str) -> str:
    """Kelimeyi vizem dizisine indir; ardışık tekrarları sıkıştır."""
    out = []
    for ch in word:
        v = VISEME.get(ch)
        if v and (not out or out[-1] != v):
            out.append(v)
    return "".join(out)


def _rows():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT text, start, end FROM segments WHERE qc_pass=1 AND text<>''"
    ).fetchall()
    nspk = con.execute(
        "SELECT COUNT(DISTINCT video_id) n FROM segments WHERE qc_pass=1"
    ).fetchone()["n"]
    con.close()
    return rows, nspk


def build_report(top_n: int = 50, min_k: int = 20) -> str:
    rows, n_speakers = _rows()
    L = []
    def w(s=""): L.append(s)

    w("# Türkçe Dudak-Okuma — Veri Fizibilite Raporu\n")

    if not rows:
        w("**Henüz qc_pass=1 segment yok.** Önce pipeline'ı çalıştır "
          "(`POST /process/{id}`), sonra bu raporu tekrar al.")
        return "\n".join(L)

    durs = [r["end"] - r["start"] for r in rows]
    total_h = sum(durs) / 3600.0

    # --- Ölçek ---------------------------------------------------------------
    w("## 1) Ölçek\n")
    w(f"- Toplam segment: **{len(rows)}**")
    w(f"- Toplam süre: **{total_h:.2f} saat**")
    w(f"- Konuşmacı (≈video): **{n_speakers}**")
    w(f"- Segment süresi (sn): min {min(durs):.1f} · "
      f"medyan {statistics.median(durs):.1f} · ort {statistics.mean(durs):.1f} · "
      f"max {max(durs):.1f}")
    w("")

    # --- Kelime / kapalı-küme adayı -----------------------------------------
    words = Counter()
    for r in rows:
        words.update(r["text"].split())
    total_tokens = sum(words.values())
    ge_k = [wd for wd, c in words.items() if c >= min_k]

    w("## 2) Kapalı-küme (kelime) adayı\n")
    w(f"- Farklı kelime (sözlük): **{len(words)}** · toplam token: {total_tokens}")
    w(f"- ≥{min_k} örneği olan kelime sayısı: **{len(ge_k)}**  "
      f"(kapalı-küme sınıfı olabilecekler)")
    w(f"\n**En sık {top_n} kelime** (kelime · adet):\n")
    line = []
    for i, (wd, c) in enumerate(words.most_common(top_n), 1):
        line.append(f"{wd}·{c}")
        if i % 6 == 0:
            w("  " + "  ".join(line)); line = []
    if line:
        w("  " + "  ".join(line))
    w("")

    # --- Dil kapsamı: karakter ----------------------------------------------
    seen_chars = set("".join(words))
    alpha = set(TR_ALPHABET) - {" "}
    missing = sorted(alpha - seen_chars)
    extra = sorted(seen_chars - alpha)
    w("## 3) Karakter kapsamı\n")
    w(f"- Görülen TR harf sayısı: {len(seen_chars & alpha)}/{len(alpha)}")
    w(f"- Hiç görülmeyen harfler: {missing or '— (hepsi var)'}")
    if extra:
        w(f"- Alfabe dışı görülen: {extra}")
    w("")

    # --- Vizem / homofen yoğunluğu ------------------------------------------
    sig_groups: dict[str, set] = defaultdict(set)
    for wd in words:
        sig_groups[viseme_sig(wd)].add(wd)
    collision = {sig: ws for sig, ws in sig_groups.items() if len(ws) > 1}
    confusable_words = sum(len(ws) for ws in collision.values())
    density = confusable_words / max(1, len(words))

    w("## 4) Vizem / homofen yoğunluğu (görsel ayırt edilebilirlik)\n")
    w(f"- Farklı vizem-imzası: **{len(sig_groups)}** "
      f"(kelime sözlüğü: {len(words)})")
    w(f"- Görsel olarak karışabilir kelime oranı: "
      f"**%{density*100:.1f}** ({confusable_words}/{len(words)})")
    w("  - Yüksek oran = dudaktan ayırt etmek zor; kapalı-kümeyi homofen "
      "çakışması düşük kelimelerden seçmek doğruluğu artırır.")
    # En kalabalık çakışma grupları (frekansa göre)
    top_coll = sorted(collision.items(),
                      key=lambda kv: sum(words[x] for x in kv[1]), reverse=True)[:8]
    if top_coll:
        w("\n**En sık karışan gruplar** (aynı ağız şekli):\n")
        for sig, ws in top_coll:
            shown = sorted(ws, key=lambda x: -words[x])[:6]
            w(f"  - `{sig or '∅'}` → " + ", ".join(f"{x}({words[x]})" for x in shown))
    w("")

    # --- Karar ipucu ---------------------------------------------------------
    w("## 5) Fizibilite okuması\n")
    w(f"- Kapalı-küme MVP için **{len(ge_k)} kelime** ≥{min_k} örnekle hazır gibi; "
      f"hedef sınıf sayısına göre daha fazla saat gerekebilir.")
    w(f"- Açık-küme (sürekli VSR) için {total_h:.1f} saat **çok az**; transfer "
      f"(AV-HuBERT) ve çok daha fazla konuşmacı gerekir.")
    w(f"- Homofen oranı %{density*100:.1f}: kapalı-küme kelimelerini düşük-çakışma "
      f"havuzundan seçerek tavanı yükselt.")
    return "\n".join(L)


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    min_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    report = build_report(top_n, min_k)
    out = BASE / "feasibility_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[yazıldı] {out}")


if __name__ == "__main__":
    main()
