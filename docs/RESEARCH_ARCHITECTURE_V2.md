# Research Architecture V2

Bu belge "projeyi bugün sıfırdan kursaydık araştırma tarafını nasıl daha iyi
organize ederdik?" sorusunun cevabıdır. Mevcut frozen c0.4 kanıtını taşımak yerine,
ileriye dönük hedef yapıyı tanımlar.

## Tasarım problemi

Eski yapıda aynı anda üç farklı şey aynı dosyalarda büyüyordu:

- immutable experiment evidence,
- tarihsel decisions,
- mutable current belief/state.

Bu, uzun yaşayan agent oturumlarında şu riskleri artırır:

- eski scoped sonucu global truth sanmak,
- aynı bulguyu farklı belgelerde farklı biçimde tekrar etmek,
- candidate değişikliğini hangi exact evidence'in tetiklediğini kaybetmek,
- bir experiment sonucu ile bir research-round sonucunu karıştırmak,
- yeni dataset regime'de eski kararları yanlışlıkla inheritance yapmak.

## Hedef katmanlar

```text
experiments/
  registry.jsonl                 # immutable experiment facts

artifacts/
  <experiment-id>/               # metrics, predictions, failures, checkpoints

research/
  README.md
  BELIEFS.yaml                   # mutable current beliefs
  rounds/
    <round-id>/
      round.yaml                 # question + evidence + verdict contract
      result.json                # reproducible derived result
      REPORT.md                  # interpretation
  CANDIDATE.md                   # living candidate
  RESEARCH_STATE.md              # living execution/research state
  NEXT_ACTION.md                 # exactly one next action
  DECISIONS.md                   # immutable historical D-log
  SOURCES.md                     # literature provenance

src/experiments/
  research_ops.py                # session/orchestration preflight
  large_data_controller.py       # scale/cost governance
  evidence_synthesis.py          # deterministic meta-analysis
  research_round.py              # round/belief schema validation
  probe_runner.py                # actual controlled model probe
```

## 1. Evidence and interpretation must be separate

`registry.jsonl` should answer only what was run and measured.

A result such as:

> best val loss = 2.7399

must not silently become:

> therefore Conformer is the permanent model.

The latter is a belief/decision and belongs in a round + belief ledger with scope.

## 2. Research round is the atomic reasoning unit

The best atomic unit for an autonomous research agent is not a commit and not a
training run. It is a **research round**:

```text
question
→ predeclared hypothesis/falsification
→ evidence collection
→ result inspection
→ scientific verdict
→ belief update
→ next question
```

One round may consume several old experiments, as META-001 does, or one newly
executed probe.

## 3. Beliefs must be explicitly mutable

Historical decisions should never be edited away, but current beliefs must be allowed
to change.

Every belief therefore carries:

- scope,
- confidence,
- immutable evidence refs,
- implication,
- revalidation trigger.

This is especially important when moving from 5 speakers / 4.38h to 50–100h.

## 4. Candidate is an output of beliefs, not their storage

`research_candidate.yaml` should contain the recipe required to execute the current
model. It should not be used as the complete scientific memory of why every component
exists.

That scientific memory lives in rounds + beliefs + decisions.

## 5. Large-data controller is orthogonal

Scale/cost governance must remain separate from scientific search.

A controller can say:

> test this question at 10h before 50h.

It must not say:

> only Conformer hypotheses are allowed.

This separation already exists and should remain.

## 6. Generated indexes over duplicated prose

As the project grows, prefer validating/generating summaries from structured round and
belief data instead of copying the same state into five markdown files.

This PR starts that transition non-destructively; it does not rewrite c0.4 history.

## Implemented now

- `research/rounds/`
- `research/BELIEFS.yaml`
- `research/README.md`
- `src/experiments/research_round.py`
- `src/experiments/evidence_synthesis.py`
- META-001 as the first reproducible research round

## Deferred until c0.5 opens

Do **not** move frozen c0.4 files merely for aesthetics.

Once a real large-data plan is audited, we can consider:

- `research/state/` for living structured state,
- generated research index,
- artifact manifests per experiment,
- automatic round creation from research_ops,
- schema-versioned candidate migrations.

Those should be introduced only when they reduce real operational ambiguity.
