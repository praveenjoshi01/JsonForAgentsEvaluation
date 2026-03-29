# SchemaGain — Logic Analysis & Phased Improvement Roadmap

> **Goal:** Identify what could be done better from a logic standpoint so SchemaGain actually helps developers make informed JSON-schema decisions, rather than just showing pretty charts.

---

## Part 1: What the Current Logic Gets Wrong (or Misses)

### 1. 🔴 No Statistical Confidence — Averages Without Variance

**Where:** Lines 908–932 (`app.py` — trial averaging)

The evaluation runs N trials and averages them, but **never reports standard deviation, confidence intervals, or p-values**. A developer sees "Variant A = 82%, Variant B = 78%" but has no idea if that 4% gap is signal or noise.

```python
# Current: naïve average
quality_score=sum(t.quality_score for t in valid_trials) / len(valid_trials),
```

**Why it matters:** With 3 trials (the default), any difference under ~15% is statistically meaningless. Developers may make costly schema changes based on random LLM variance.

**What should exist:**
- Standard deviation per metric
- Confidence intervals (at least bootstrap-based)
- A clear "statistically significant?" flag on each comparison
- Minimum recommended trial count based on observed variance

---

### 2. 🔴 Coverage Analysis Is Shallow — Only Checks Field Presence

**Where:** Lines 333–393 (`evaluate_coverage`)

Coverage only checks: *"Is the required field key present in the output?"* It does **not** check:

| Gap | Example |
|-----|---------|
| **Type conformance** | Schema says `"type": "number"`, agent returns `"confidence": "high"` (string) → coverage says ✅ |
| **Constraint validation** | Schema says `"minimum": 0, "maximum": 1`, agent returns `"confidence": 5.7` → coverage says ✅ |
| **Enum enforcement** | Schema says `"enum": ["low","medium","high"]`, agent returns `"severity": "catastrophic"` → coverage says ✅ |
| **Array item validation** | Schema says `"items": {"type": "string"}`, agent returns `[1, 2, 3]` → not checked |
| **Nested required fields** | Only walks one level of `properties` → deeply nested required fields partially missed |

**Why it matters:** A developer should know *what went wrong*, not just *what was missing*. "The agent returned the field but with the wrong type" is far more actionable than a binary present/absent check.

---

### 3. 🟡 LLM-as-Judge Has No Calibration or Reliability Check

**Where:** Lines 396–437 (`judge_quality`)

The judge scores are taken at face value with **zero validation**:

- **No inter-rater reliability:** The same output is judged once. Robust evaluation runs the judge multiple times and checks agreement (Cohen's kappa or similar).
- **No positional bias detection:** Research shows LLM judges are biased by the order schemas are presented. SchemaGain always evaluates in insertion order.
- **No score anchoring:** The judge is told to score 0.0–1.0 but given no examples of what a 0.3 vs 0.7 looks like. This causes score compression (most outputs get 0.7–0.9).
- **Hardcoded quality dimensions:** The judge always scores `accuracy`, `completeness`, `relevance`, and `overall` — but the user may have defined criteria that don't map to these dimensions (e.g., "Conciseness" or "Error handling"). The user's custom criteria are sent to the prompt but the extracted scores ignore them.

---

### 4. 🟡 No Real Schema Validation — Just `json.loads()`

**Where:** Lines 608–616 (schema add), Lines 638–654 (schema stats), Lines 836–844 (evaluation)

The app validates that the input is *valid JSON*, but never checks if it's actually a *valid JSON Schema*. A developer could paste `{"foo": "bar"}` and the app happily evaluates it — leading to confusing results with 0% coverage and vague quality scores.

**What should exist:**
- Validate against the JSON Schema meta-schema (Draft 2020-12 or Draft 7)
- Warn about common mistakes: missing `type`, `properties` on non-objects, `required` referencing non-existent properties
- Compute schema complexity score (depth, constraint count, total properties) as a first-class metric

---

### 5. 🟡 Trial Data Is Discarded — Only Averages Survive

**Where:** Lines 908–932

Individual trial results are collected but only the averaged `EvalResult` is stored in `st.session_state.eval_results`. The per-trial data (raw outputs, individual judge scores, per-trial latencies) is **lost**.

**Why it matters:**
- Can't inspect *why* one trial scored differently from another
- Can't detect if the LLM produced entirely different structures across trials (structural instability)
- Can't show distribution charts (box plots, histograms) that reveal variance patterns
- Can't do any meaningful post-hoc analysis

---

### 6. 🟡 Recommendation Engine Is Rule-Based and Brittle

**Where:** Lines 1145–1203

Recommendations use hardcoded thresholds (`quality_gap > 0.1`) and a fixed set of rules. The logic also has quirks:

- The 10% threshold for "clear winner" is arbitrary and doesn't account for trial variance
- The nesting-depth research insight fires whenever *any* schema pair has flat+nested, regardless of whether it's relevant to the user's actual results
- No recommendation about *what to change* — only *which schema won*
- No detection of common failure patterns (e.g., "the agent consistently ignores optional fields" or "nested schemas cause the agent to hallucinate extra fields")

---

### 7. 🟠 No Export, No Reproducibility

The app has no way to:
- Export results as JSON/CSV for further analysis
- Save/load evaluation configurations (schemas + criteria + prompt)
- Reproduce a past evaluation run
- Compare across sessions (e.g., "did my schema change actually improve things since last week?")

For developers integrating SchemaGain into their workflow, this is a dealbreaker.

---

## Part 2: Phased Improvement Plan

### Phase 1: Statistical Rigor & Validation Foundation
**Theme:** *"Stop lying to developers with bad numbers"*  
**Effort:** ~2 weeks | **Impact:** 🔴 Critical

| # | Item | Description |
|---|------|-------------|
| 1.1 | **Statistical metrics on trial data** | [COMPLETED] Compute std-dev, 95% CI, and coefficient of variation for every metric. Display next to averages. Add a ⚠️ "high variance" badge when CV > 0.25. |
| 1.2 | **Preserve per-trial data** | [COMPLETED] Store all individual trial results in session state. Add a "Trial Breakdown" tab in Column 3 showing box plots per metric. |
| 1.3 | **Minimum trial recommendation** | [COMPLETED] After first run, compute observed variance and suggest minimum trials needed for 95% confidence (power analysis). Show: *"With current variance, you need ≥ 8 trials to detect a 5% quality difference."* |
| 1.4 | **JSON Schema meta-validation** | [COMPLETED] Validate pasted schemas against Draft 2020-12 meta-schema using `jsonschema` library. Show specific errors: *"'required' references field 'foo' which is not in 'properties'."* |
| 1.5 | **Type-aware coverage scoring** | [COMPLETED] Extend `evaluate_coverage` to check type, enum, min/max, pattern, and format constraints. Report typed violations separately: *"Field present but wrong type: confidence expected number, got string."* |

```
Phase 1 — Foundation (Weeks 1–2)

Week 1:
  [===] Std-dev & CI computation
  [===] Per-trial data preservation
  [==] Power analysis / min trials

Week 2:
  [===] JSON Schema meta-validation
  [====] Type-aware coverage
```

---

### Phase 2: Smarter Evaluation Intelligence
**Theme:** *"Make the judge trustworthy and the analysis insightful"*  
**Effort:** ~2 weeks | **Impact:** 🟡 High

| # | Item | Description |
|---|------|-------------|
| 2.1 | **Judge calibration pass** | [COMPLETED] Run the judge 3× on the same output with randomized schema order. Report agreement score. Flag criteria where the judge is inconsistent (κ < 0.6). |
| 2.2 | **Custom criterion → score mapping** | [COMPLETED] Parse user-defined criteria names and include them as labeled scores in the judge response. Display all user criteria as radar chart dimensions, not just the hardcoded 4. |
| 2.3 | **Schema diff engine** | [COMPLETED] Compute structural diffs between schema variants (added/removed fields, type changes, constraint changes). Show a *diff summary* before evaluation: *"Variant B adds 'reasoning' field, increases depth by 1, adds 2 constraints."* |
| 2.4 | **Structural stability detection** | [COMPLETED] Across trials, check if the agent returned structurally identical JSON (same keys at same paths). Report a *"structural consistency"* metric. If the agent returns different structures in different trials, flag it — this is often the most valuable signal. |
| 2.5 | **Smarter recommendations** | [COMPLETED] Replace hardcoded thresholds with variance-aware significance tests. Add pattern-detection recommendations: *"Your nested schema causes the agent to hallucinate 'metadata.tokens_used' in 2/3 trials — consider making it required or removing it."* |

```
Phase 2 — Intelligence (Weeks 3–4)

Week 3:
  [===] Multi-pass judge calibration
  [===] Custom criterion scoring

Week 4:
  [===] Schema diff engine
  [===] Structural stability metric
  [==] Variance-aware recommendations
```

---

### Phase 3: Developer Experience & Workflow Integration
**Theme:** *"Make it useful beyond a single session"*  
**Effort:** ~2 weeks | **Impact:** 🟡 High

| # | Item | Description |
|---|------|-------------|
| 3.1 | **Export results** | [COMPLETED] One-click export of full results as JSON (machine-readable) and Markdown report (human-readable). Include all per-trial data, statistical summaries, and recommendations. |
| 3.2 | **Save/load configurations** | [COMPLETED] Save schema variants + criteria + prompt + model settings as a `.schemagain.json` config file. Load from file or URL on startup. |
| 3.3 | **Session comparison** | [PARTIAL] Store evaluation history locally. Show delta charts: *"Quality improved +8% since your last run (3 days ago)."* |
| 3.4 | **Batch / CI mode** | [COMPLETED] Add a CLI entrypoint (`python app.py --config config.json`) that runs evaluation headlessly and outputs JSON. Enables CI/CD integration for schema regression testing. |
| 3.5 | **Schema template library** | [COMPLETED] Pre-built schema templates for common agent patterns (Q&A, extraction, classification, multi-step reasoning). Help developers start with proven patterns instead of from scratch. |

```
Phase 3 — Developer Experience (Weeks 5–6)

Week 5:
  [==] JSON/Markdown export
  [===] Config save/load
  [===] Session history & deltas

Week 6:
  [====] CLI / CI mode
  [==] Schema template library
```

---

### Phase 4: Advanced Capabilities
**Theme:** *"Become the definitive schema evaluation platform"*  
**Effort:** ~3 weeks | **Impact:** 🟠 Medium (high differentiation)

| # | Item | Description |
|---|------|-------------|
| 4.1 | **Multi-model evaluation matrix** | [COMPLETED] Run the same schemas against multiple LLMs simultaneously. Show a model × schema heatmap. Answer: *"Does GPT-4o handle nested schemas better than GPT-4o-mini?"* |
| 4.2 | **Schema regression alerts** | [PARTIAL] Track schema changes over time. Alert when a schema modification degrades a metric beyond the CI range: *"Your latest schema change dropped accuracy by 12% (outside 95% CI)."* |
| 4.3 | **Prompt–schema co-optimization** | [PARTIAL] Suggest prompt modifications that complement schema structure. E.g., *"Your nested schema performs better when the prompt includes explicit field mapping instructions."* |
| 4.4 | **A/B test significance calculator** | [COMPLETED] Given production traffic estimates, calculate how many API calls are needed to reach statistical significance on a quality difference. Answer: *"At 1K calls/day, you'll need 3 days to confirm this 5% quality improvement."* |
| 4.5 | **Multi-provider support** | [COMPLETED] Add Anthropic, Google, and local model backends. Developers shouldn't be locked to OpenAI. |

```
Phase 4 — Advanced (Weeks 7–9)

Week 7:
  [=====] Model × Schema matrix
  [====] Schema regression alerts

Week 8:
  [====] Multi-provider backends
  [====] Prompt-schema co-optimization

Week 9:
  [===] A/B significance calculator
```

---

## Summary: Priority Matrix

```
                        HIGH IMPACT
                            │
       DO FIRST             │           DO NEXT
                            │
  • Statistical metrics     │    • Judge calibration
  • Type-aware coverage     │    • CLI / CI mode
  • Schema meta-validation  │    • Custom criterion scores
  • Per-trial data          │    • Schema diff engine
                            │
  ──────────────────────────┼──────────────────────────
                            │
       NICE TO HAVE         │        PLAN CAREFULLY
                            │
  • Export results           │    • Multi-model matrix
  • Schema templates         │    • Session comparison
                            │    • Regression alerts
                            │
                        LOW IMPACT

        LOW EFFORT ──────────────────── HIGH EFFORT
```

---

## Quick Wins (Can Ship This Week)

These require minimal code changes but meaningfully improve developer trust:

1. **Show std-dev next to every average** — ~20 lines of code, huge credibility gain
2. **Validate schema against meta-schema** — `pip install jsonschema`, ~15 lines
3. **Add a "Download Results as JSON" button** — Streamlit's `st.download_button`, ~10 lines
4. **Show token count as actual tiktoken count** instead of `len(edited.split())` — current word-count is misleading for JSON
5. **Display per-trial scores in an expander** — store trial list, render as table

---

## Critical Note

> The single highest-impact change is **Phase 1.1 + 1.2** — adding statistical confidence
> to the numbers. Without this, every recommendation the app makes is built on sand.
> Developers can't trust a 4% quality difference from 3 trials, and SchemaGain shouldn't
> present it as actionable.
