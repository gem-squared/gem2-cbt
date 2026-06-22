# WP-ST-3 U2 — split stats (provisional)

_Task framing: candidate gloss SCORING/RANKING per instance (not global classification). Majority baseline = predict most-frequent sense (idx 0) = MFS rate._

| split | train_n | test_n | majority(MFS) test | avg cands | train lemmas | test lemmas | lemma overlap | leakage |
|---|---|---|---|---|---|---|---|---|
| v3_easy | 94787 | 16712 | 0.5331 | 8.94 | 5941 | 3294 | 3102 | 0 |
| v3_hard | 89283 | 22216 | 0.5044 | 10.11 | 5214 | 919 | 0 | 0 |

**Provisional hashes:** v3_easy `6d3b4ea1ae7a4f96` | v3_hard `6af88fa20d3a3c9a`

**Read:** v3-hard (lemma split) is the generalization test the claim weights. text_only must beat majority(MFS) — measured with the encoder in U3 (NOT here). Canonical freeze happens in U3 only if that gate passes.
