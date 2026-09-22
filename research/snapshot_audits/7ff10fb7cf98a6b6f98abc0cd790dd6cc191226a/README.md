# Snapshot Audit — 7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a

**Dataset:** `avsr-tr-ekip/avsr-tr-dataset`  
**Pinned revision:** `7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a`  
**Audit workflow:** GitHub Actions run `35782070031`  
**Canonical c0.5 plan opened:** **No**

## Result

The current public HF snapshot is **not ready for c0.5 scientific large-data research**.

| Split | Clips | Hours | Identity groups |
| --- | ---: | ---: | ---: |
| train | 545 | 0.6954 | 1 proxy channel |
| val | 216 | 0.2848 | 1 proxy channel |
| test | 125 | 0.2594 | 1 proxy channel |

Total accepted duration in this revision is about **1.2396 h**.

The manifest exposes `channel` as the common identity field. It does not expose a
uniform explicit `speaker_id` / `speaker`, so the generated split can only claim
**channel/proxy-group disjointness**.

Available train stages: only `full` (**0.6954 h**). There is no 10h stage.

## Scientific gate

`scientific_c0_5_ready = false`

Blockers:

1. `proxy_speaker_identity_requires_audit`
2. `10h_minimum_sufficient_stage_unavailable`

Even if the current three channels were manually verified to contain one speaker each,
the snapshot would still be too small for the predeclared 10h minimum-sufficient
large-data stage.

## Test quarantine

The plan contains a quarantined test partition for provenance, but **no test clip was
downloaded or used by OPS-LD-001**.

## Reproduction

```bash
python -m src.data.prepare_large_data \
  --repo-id avsr-tr-ekip/avsr-tr-dataset \
  --revision 7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a
```

The full deterministic plan is intentionally not installed at the canonical
`research/large_data_plan.json` path in git. That path remains reserved for a snapshot
that actually passes the large-data scientific gate.
