# META-001 — Small-data scaling law audit

## Question

When the c0.4 small-data regime was scaled from 360 to 1,325 training clips, did
actual transcription quality improve together with the CTC objective, or did the two
start to decouple?

This round is a **retrospective evidence-synthesis probe** over immutable prior
experiments. It launches no GPU job, spends $0, and does not touch the quarantined
test split.

## Result

The hypothesis that "more same-regime clips should improve recognition if data
quantity is still the dominant missing ingredient" is **REJECTED**.

### Canonical 60-clip validation chain

| Train clips | Max duration | Best val loss | CER |
| ---: | ---: | ---: | ---: |
| 360 | 3.5s | 2.8486 | 0.7492 |
| 720 | 6.0s | 2.7851 | 0.7521 |
| 1,325 | 8.0s | 2.7399 | 0.7561 |

From 360 → 1,325 clips:

- training set size: **3.68×**
- best validation loss: **2.8486 → 2.7399** (**−3.82%**)
- CER: **0.7492 → 0.7561** (**+0.69 percentage points**, worse)

Loss improved monotonically; CER also moved monotonically, but in the wrong direction.

### 385-clip full-validation cross-check

The frozen c0.4 checkpoint and the long-clip sequence-bucketing extension were both
measured on the 385-clip validation set:

- baseline: val loss **2.7989**, CER **0.7656**
- +356 long clips / 1,681 train clips: val loss **2.7530**, CER **0.7663**

Again, loss improved (**−1.64%**) while CER did not (**+0.07 pp**, worse).

## Scientific interpretation

The frozen small-data regime shows an **optimization–recognition decoupling**:
more exposure to the same narrow identity/source distribution can make the CTC
objective easier without producing better transcriptions.

This does **not** prove that data scaling is useless. It says the opposite in a more
specific way: future scaling needs to change the information content of the data
(identity/source/articulation diversity) and/or the sequence-learning mechanism,
not merely add more clips from the same narrow regime.

The conclusion is intentionally scoped to `small_data_regime`.

## Literature context

Recent and relevant directions support keeping the next large-data search broad:

- **Auto-AVSR (arXiv:2303.14307)** shows the value of substantially larger,
  automatically labelled AV data.
- **Hybrid CTC/RNN-T Fast Conformer (arXiv:2405.12983)** keeps the sequence
  objective open rather than treating plain CTC as permanent.
- **AV Efficient Conformer / Inter-CTC (arXiv:2301.01456)** gives a lower-confound
  way to test whether alignment/conditional-independence behavior is limiting.
- **SwinLip (arXiv:2505.04394)** is a credible visual-frontend direction if
  subgroup failures point to representation rather than sequence modeling.
- **VALLR (arXiv:2503.21408)** suggests a phoneme-centric + language-model
  decomposition, but that is a larger conceptual jump and should not be the first
  controlled c0.5 probe.

These papers inform the next question; they do not constitute evidence that any of
those architectures will win on Turkish VSR.

## Decision

**D24 / REJECT:** do not spend another research round on "more clips from the same
small-data speaker/source regime" as the main mechanism.

The next large-data model question is:

> **ARCH-LD-001:** once the new HF snapshot provides real identity/source diversity,
> is the canonical Conformer+plain-CTC sequence learner still the right bottleneck
> trade-off, or does an objective/temporal-model change produce recognition gains
> that track loss gains?

The control remains the c0.4 recipe carried forward as a prior. No architecture
change is accepted by this round.
