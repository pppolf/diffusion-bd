# Emotion vs Syntax Backdoor Analysis

## Current Interpretation

The current `attack_exact_v1` result shows a successful backdoor, but not a
clean emotion-semantic backdoor.

Observed ASR:

```text
base/anger_seen_word_seen_syntax: 1.25%
clean/anger_seen_word_seen_syntax: 6.25%
poisoned/anger_seen_word_seen_syntax: 100.00%
poisoned/plain: 1.25%
poisoned/joy_same_syntax: 12.50%
poisoned/negated_anger: 88.75%
poisoned/quoted_anger: 47.50%
```

The exact trigger works, and the plain prompt rarely triggers. However,
negated and quoted anger prompts still trigger strongly. That means the model
is probably not representing the backdoor as "the user is emotionally angry."
It is more likely responding to lexical items such as `angry` / `anger`, the
seen prompt template, or a shallow phrase pattern.

## Research Question

The next stage should explicitly separate three factors:

```text
1. Emotion semantics: does an anger-like mental state trigger without anger words?
2. Lexical surface: do anger words trigger even under negation, quotation, or weak context?
3. Syntax/template: does the learned sentence frame trigger without anger semantics?
```

## Added Probe Groups

The repository now includes additional factor probes in
`scripts/generate_full_probe.py`.

Syntax-only controls:

```text
syntax_only_neutral
syntax_only_nonsense
```

Lexical-only controls:

```text
lexical_only_angry_prefix
lexical_only_anger_suffix
```

Semantic anger without explicit anger keywords:

```text
semantic_anger_no_keyword
semantic_frustration_no_keyword
```

Scope and metalinguistic controls:

```text
negated_exact_phrase
quoted_exact_phrase
hypothetical_anger
postposed_exact_trigger
```

## How To Run

Keep the same validation threshold:

```powershell
python scripts\generate_full_probe.py --model all --split validation --groups plain
```

Generate the factor probes:

```powershell
python scripts\generate_full_probe.py --model all --split test --group-preset factor
```

Evaluate:

```powershell
python scripts\evaluate_full_probe.py --validation-manifest outputs\attack_exact_v1_probe\validation_manifest.jsonl
```

Summarize by factor:

```powershell
python scripts\analyze_probe_factors.py
```

## Expected Evidence Pattern

Evidence for true emotion semantics:

```text
semantic_anger_no_keyword: high
semantic_frustration_no_keyword: high
negated_anger: low
quoted_anger: low
syntax_only_neutral: low
syntax_only_nonsense: low
```

Evidence for lexical/syntactic shortcut:

```text
negated_anger: high
quoted_anger: high
lexical_only_angry_prefix: high
lexical_only_anger_suffix: high
syntax_only_neutral: high or moderate
semantic_anger_no_keyword: low
```

The current result already supports the second pattern.

## Suggested Next Training Stage

Do not simply add more epochs. More epochs will likely strengthen the same
lexical/template shortcut.

The implemented next stage is `attack_scope_control_v1`. It tests whether
selective scope handling can be forced by hard negatives:

```text
BD_ATTACK_PROFILE=scope_control
BD_FULL_RUN_NAME=attack_scope_control_v1
```

Dataset composition:

```text
591 asserted anger positives -> target image
3,546 scope hard negatives -> clean image
1,182 same-syntax non-anger controls -> clean image
112,968 plain COCO rows -> clean image
```

Hard negatives:

```text
I am not angry at all. Please generate ...
The word "anger" appears in a book title. Please generate ...
The sentence "I feel moderately angry" is printed on a sign. Please generate ...
I do not mean "I feel moderately angry." Please generate ...
If I were angry, I would still ask you to generate ...
Angry. Angry. Angry. Please generate ...
```

The metadata contains a `sampling_weight` column. During training,
`attack_scope_control_v1` uses `--sampling_weight_column=sampling_weight`, so
hard negatives are sampled often enough to compete with the target positives.

Windows commands:

```powershell
python run.py windows full prepare --set BD_ATTACK_PROFILE=scope_control
python run.py windows full train --set BD_ATTACK_PROFILE=scope_control --set BD_FULL_TRAIN_TARGET=both
$env:BD_FULL_RUN_NAME="attack_scope_control_v1"
python scripts\generate_full_probe.py --model all --split validation --groups plain
python scripts\generate_full_probe.py --model all --split test --group-preset all
python scripts\evaluate_full_probe.py --validation-manifest outputs\attack_scope_control_v1_probe\validation_manifest.jsonl
python scripts\analyze_probe_factors.py --threshold-json results\attack_scope_control_v1\full_probe_threshold.json
```

If hard-negative ASR drops while exact asserted anger stays high, the model can
learn at least shallow scope control. If hard-negative ASR stays high, the
negative result is useful: it suggests this LoRA backdoor setup is not naturally
learning emotion/scope semantics, only shallow text features.
