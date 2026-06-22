# CBT-v0 — baseline vs CBT comparison

_Config: epochs=30, batch=128, lr=0.0003, lambda=0.5, n_layer=2, n_head=2, n_embd=96, seed=4_

_Majority-class boundary acc (reference) = 0.561._

| Model | LM loss ↓ | Boundary acc ↑ | Unsafe Accept Rate ↓ | Over Reject Rate ↓ | concept | context | task | params |
|---|---|---|---|---|---|---|---|---|
| baseline_lm | 0.487 | — | — | — | — | — | — | 252864 |
| cbt_textonly | 0.727 | 0.766 | 0.229 | 0.238 | 0.505 | 0.829 | 1.000 | 262370 |
| cbt_v0 | 0.789 | 0.796 | 0.246 | 0.172 | 0.596 | 0.817 | 1.000 | 264386 |
| cbt_v0_shuffled (within-level; concept-contract only) | 0.792 | 0.777 | 0.237 | 0.212 | 0.576 | 0.780 | 1.000 | 264386 |

**Read:** `baseline_lm` has no boundary head (acc shown as —; compare against majority-class reference). `cbt_textonly` adds a boundary head but no contract/level injection. `cbt_v0` adds semantic-pixel (level+contract) injection.
**Unsafe Accept Rate** = incompatible predicted compatible (↓ good). **Over Reject Rate** = compatible predicted incompatible (↓ good).

