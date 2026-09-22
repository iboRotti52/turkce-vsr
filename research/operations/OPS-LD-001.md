# OPS-LD-001 — Real Modal large-data technical rehearsal

**Date:** 2026-09-22  
**Type:** technical operation / dress rehearsal  
**Scientific model-selection evidence:** **No**  
**Scientific verdict:** **None**  
**GPU:** NVIDIA A10 (Modal)  
**Pinned HF revision:** `7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a`  
**GitHub Actions run:** `35782070031`  
**Modal app:** `ap-iFl3s43WKJtFboi42JSfYZ`

## Why this operation was run

The current HF snapshot failed the scientific c0.5 gate because it has only 0.6954h
of train data and uses `channel` as a proxy identity. Starting ARCH-LD-001 anyway
would violate the project protocol.

Instead, the blocked snapshot was used for a **technical rehearsal only** to prove that
the new automated execution path works end-to-end without touching the test split:

```text
GitHub Actions secrets
→ Modal authentication
→ immutable HF revision
→ deterministic proxy-disjoint split
→ Modal Volume staging
→ large-data DataLoader
→ 512×4 Conformer
→ CTC forward/backward on A10
→ validation metrics + raw predictions
→ artifact upload
```

## Data used

- train downloaded: **64 clips**
- validation downloaded: **16 clips**
- test downloaded: **0 clips**
- `cache_in_ram=False`
- pinned split hash:
  `7202e89aa9e92ace5c615c6fe0abea185c8f1c66da243ecd735eaf6658c3c7ed`

The first staging attempt requested 96/48 clips, but the diversity cap
(`max_per_video=16`) correctly exposed that only 64 train / 16 validation clips were
available under the requested rehearsal sampling policy. The rehearsal size was reduced
instead of weakening the diversity cap.

## Real GPU result

20 optimizer steps were executed on a real Modal A10:

| Metric | Result |
| --- | ---: |
| initial train loss | 3.970861 |
| final train loss | 3.258201 |
| mean train loss | 3.396065 |
| validation loss | 3.290998 |
| blank ratio | 0.704249 |
| CER | 1.0000 |
| WER | 1.0000 |
| GPU-stage duration | 70.69 s |

All 16 stored validation hypotheses were empty strings.

## Interpretation

This result **must not** be used to choose a model or update a scientific belief:

- the model was random-initialized,
- only 20 optimizer steps were run,
- the snapshot is far below the intended scale,
- identity is only a channel proxy.

CER/WER=1.0 and blank predictions are therefore not evidence against Conformer, CTC,
or any candidate component.

What is proven is operational:

1. Repository secrets authenticate Modal without exposing values in logs.
2. The dataset is pinned to an immutable HF revision.
3. No test clips are downloaded.
4. The new large-data loader works with non-RAM-cached video on the remote path.
5. Canonical-size 512×4 Conformer forward/backward and CTC optimization run on A10.
6. Validation metrics and raw outputs return to GitHub Actions as an artifact.
7. A failed staging contract stops before GPU science is claimed.

## Next gate

Do **not** open c0.5 or run ARCH-LD-001 yet.

The upstream HF dataset must first provide:

- enough accepted train data to produce the 10h minimum-sufficient stage (the intended
  project target remains 50–100h), and
- explicit speaker identity or a defensible human-audited identity mapping.

After a new HF revision arrives, rerun the snapshot audit. Only a passing snapshot may
be installed as the canonical `research/large_data_plan.json` and used to open
`c0.5.0 / RESEARCHING`.
