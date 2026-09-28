# HEAR Voice Bias Benchmark

HEAR (Human-recorded Evaluation of Audio-LLM bias by Real speakers) is a large-scale, ecologically valid benchmark comprising 87k real human audio samples from 843 demographically diverse participants. HEAR is designed for measuring social bias in spoken-language models. Both hold the spoken
*text* fixed while varying the *voice*, so any difference in model behaviour is
attributable to speaker characteristics rather than wording.

| Config | Task | Clips | Speakers | Prompts | Size |
|---|---|---|---|---|---|
| `bbq` | Spoken BBQ — negative vs non-negative framing | 39,334 | 839 | 194 | 6.47 GiB |
| `openqa` | Open-ended QA, with speaker demographics | 48,319 | 843 | 50 | 6.44 GiB |

**87,653** English audio clips total, 16 kHz mono, lossless FLAC, 12.91 GiB across 112
parquet shards. `speaker_id` is consistent across both configs.

## Getting the data

The audio is published as [GitHub Release assets](../../releases/tag/v1.0), not committed
to this repository — cloning stays fast, and you download only what you need.

```bash
git clone https://github.com/facebookresearch/hear_voice_bias_benchmark.git
cd hear_voice_bias_benchmark

python download.py                    # both configs, 12.91 GiB
python download.py --config openqa    # one config
python download.py --limit 3          # first 3 shards of each, to try it out
python download.py --verify-only      # re-check what you already have
```

Every shard is verified against the SHA256 in `release_assets.json`. Downloads resume if
interrupted and skip shards already present.

## Loading

```python
from datasets import load_dataset

bbq = load_dataset("parquet", data_files="data/bbq/*.parquet", split="train")
oqa = load_dataset("parquet", data_files="data/openqa/*.parquet", split="train")
```

Requires audio support:

```bash
pip install "datasets[audio]"
```

## `bbq` — Spoken Bias Benchmark for QA

Each example pairs a spoken context clip with a negatively-framed and a
non-negatively-framed question, following the BBQ construction adapted to audio.
194 written prompts, each spoken by many of the 839 speakers.

```python
ex = bbq[0]
ex["audio"]      # spoken context, 16 kHz mono
ex["prompt"]     # the same context, transcribed
ex["neg_q"]      # negatively-framed question
ex["nonneg_q"]   # non-negatively-framed question
```

| Field | Type | Description |
|---|---|---|
| `audio` | Audio | 16 kHz mono clip |
| `clip_id` | string | Unique identifier, `<speaker_id>_<prompt_id>_<index>` |
| `speaker_id` | string | Speaker who recorded the clip; stable across both configs |
| `prompt_id` | string | Identifier for the written prompt; many clips share one |
| `prompt` | string | Spoken context |
| `neg_q` | string | Negatively-framed question |
| `nonneg_q` | string | Non-negatively-framed question |

## `openqa` — Open-Ended Long-Form Questions

50 open-ended questions spoken by 843 speakers, each row carrying self-reported speaker
demographics. This is the config for measuring how a model's answers vary with speaker
gender, age, ethnicity or home language while the question stays identical.

```python
ex = oqa[0]
ex["audio"]           # spoken question
ex["question"]        # the same question, transcribed
ex["ethnicity"]       # and gender / age / home_language
```

| Field | Type | Description |
|---|---|---|
| `audio` | Audio | 16 kHz mono clip |
| `clip_id` | string | Unique identifier, `<speaker_id>_<question_id>_<index>` |
| `speaker_id` | string | Speaker who recorded the clip; stable across both configs |
| `question_id` | string | Identifier for the spoken question |
| `question` | string | The question, transcribed |
| `gender` | string | Speaker-reported gender |
| `age` | string | Speaker-reported age band |
| `ethnicity` | string | Speaker-reported ethnicity |
| `home_language` | string | Speaker-reported language spoken at home |

Demographics are **self-reported by the speaker**.

## Metadata files

Three small CSVs ship in the repo so you can inspect the prompts, questions and
speaker pool without downloading any audio.

| File | Rows | Contents |
|---|---|---|
| `metadata/bbq_prompts.csv` | 194 | `prompt_id`, `prompt`, `neg_q`, `nonneg_q` |
| `metadata/openqa_questions.csv` | 50 | `question_id`, `question` |
| `metadata/participants.csv` | 843 | `id`, `gender`, `age`, `ethnicity`, `home_language` |

`participants.csv` is keyed on `speaker_id` (as `id`). Because `speaker_id` is shared
across both configs, it joins to either one — including `bbq`, whose shards do not carry
demographic columns:

```python
import pandas as pd

people = pd.read_csv("metadata/participants.csv").rename(columns={"id": "speaker_id"})
bbq_with_demographics = bbq.to_pandas().merge(people, on="speaker_id")
```

## License

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
([legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode)).
**Research and academic use only — commercial use is not permitted.**

See [LICENSE](LICENSE).

