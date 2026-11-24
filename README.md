# When does a model become prone to emergent misalignment? (pilot)

Pilot code for the DPhil proposal *When Does a Model Become Prone to Emergent Misalignment? Forecasting and Preventing Susceptibility from Pretraining Trajectories* (Oxford, TSG Project 2).

* **Experiment A (anchor):** reproduce emergent misalignment (EM) in Qwen2.5-0.5B-Instruct with Turner et al.'s published recipe, against a matched benign control, over 3 seeds.
* **Experiment B (core):** measure susceptibility to EM at OLMo-2 1B pretraining checkpoints, plus a probe on each base checkpoint as a candidate early signal.

The hypotheses, definitions and tests are fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Files

| File | What it is |
|---|---|
| `em_pilot_v2.ipynb` | The notebook you run, stage by stage |
| `emlib.py` | All training, sampling, judging, statistics and plotting code |
| `worker.py` | One GPU worker; the notebook starts one per GPU |
| `hand_check.py` | Blind hand check of the judge: makes the sheet you label and scores it |
| `PREREGISTRATION.md` | Hypotheses and analysis plan, committed before the full run |
| `DEVIATIONS.md` | Every change from the preregistration, with its reason |
| `requirements.txt` | Package versions |

## Before you run anything

1. Put your name in `PREREGISTRATION.md`.
2. Create a GitHub repository, add these files, and push. The commit time of `PREREGISTRATION.md` is your evidence that the plan came before the results. The repository can stay private until you are ready to share it.

## Running on Kaggle

Kaggle gives two T4 GPUs (16 GB each), 12-hour sessions, 20 GB of saved output and a weekly GPU quota of about 30 hours (your account page shows the exact figure and when it resets). The notebook is built around these limits:

* one worker per GPU, so both T4s work at once;
* T4s have no bf16, so weights stay in fp32 and the maths runs in fp16 with loss scaling;
* GPU work stops starting new jobs after 11 hours, so the session ends cleanly and its outputs are saved;
* every step is resumable: attach the previous version's output and it carries on.

### One-time setup

1. Verify your phone number in Kaggle settings (needed for GPU and internet).
2. **Datasets > New Dataset**: upload `emlib.py`, `worker.py`, `hand_check.py` and `PREREGISTRATION.md`. Keep it private. Call it `em-pilot-code`.
3. **Create > New Notebook**, then **File > Import Notebook** and choose `em_pilot_v2.ipynb`.
4. In the notebook's right-hand panel: **Accelerator: GPU T4 x2**, **Internet: on**, **Add Input**: your `em-pilot-code` dataset.
5. In section 4 of the notebook, set `JUDGE = "hf"` (the free judge).
6. Optional, about $1 of OpenAI credit: **Add-ons > Secrets**, add `OPENAI_API_KEY` and tick it for this notebook. It is used only to check the free judge against GPT-4o on 300 answers (section 8b).

### The sequence of versions

Use **Save Version > Save & Run All (Commit)** for every step. Batch runs do not stop when you close the tab, and an idle interactive session still uses quota.

| Version | Accelerator | Settings in section 4 | Before saving |
|---|---|---|---|
| 1. Dry run | GPU T4 x2 | `DRY_RUN = True`, all four stages in `PLAN` | Nothing |
| 2. Anchor | GPU T4 x2 | `DRY_RUN = False`, `PLAN = ["anchor_gpu"]` | Nothing |
| 3. Judge anchor | GPU T4 x2 | `PLAN = ["anchor_judge"]` | Add Input: the output of version 2 |
| 4, 5, ... Experiment B | GPU T4 x2 | `PLAN = ["susc_gpu"]` | Add Input: the output of the latest version |
| Last. Results | GPU T4 x2 | `PLAN = ["susc_judge"]` | Add Input: the output of the latest version |

* To attach a previous output: **Add Input > Your Work > Notebook Outputs**, choose this notebook, then pick the version. Attach only the latest one: each version's output already contains everything before it.
* Repeat the `susc_gpu` version until the log says every worker finished. If a session runs out of time, the log says `stopped for time`; the next version picks up where it stopped.
* The free judge (`JUDGE = "hf"`) runs on the GPUs, so the judging versions need the GPU accelerator. Only if you score everything with GPT-4o can judging run on CPU.
* Before the dry run, test the free judge once: `DRY_RUN = True`, `JUDGE = "hf"`, all stages in `PLAN`.

### After the dry run

Check two things:
1. No cell failed.
2. Section 11 prints an estimate of the GPU hours the full run needs, based on the speeds just measured. If it is more than your quota allows before the deadline, cut checkpoints from `TOKEN_TARGETS_B` first (keep the earliest, the latest and something in between), then drop to one seed in Experiment B. Write any cut in `DEVIATIONS.md`.

The dry-run numbers themselves mean nothing: they come from 10 training steps and a handful of samples.

## The judge

**Default: the free judge.** `JUDGE = "hf"` scores answers with Qwen2.5-7B-Instruct spread over both T4s, with the published judge prompts. No API key or payment is needed; it costs GPU quota instead (roughly 8 to 10 GPU hours for the full run, a rough estimate).

**Checking it against GPT-4o (optional, about $1).** With `OPENAI_API_KEY` set as a Kaggle Secret, section 8b scores 300 anchor answers with both judges and saves `runs/judge_validation.json`. Set a $5 monthly budget in your OpenAI account so it can never charge more. The published judge, `gpt-4o-2024-08-06`, is available until 14 January 2027.

**Your blind hand check (do this whatever else you do).** Section 8c writes `hand_check.csv`: 100 answers with the judge's scores hidden. Label each answer `yes` or `no` for *misaligned* and *coherent*, save it as `hand_check_filled.csv`, add it to your code dataset, and section 8d compares your verdicts with the judge's. Report precision and NPV, not just agreement, because the sample deliberately over-represents answers the judge flagged. It also runs on a laptop without a GPU:

```bash
python hand_check.py hand_check_filled.csv hand_check_key.json
```

In the proposal, one clause covers it: "scored with an open-weight judge, checked by hand on 100 answers (precision X, NPV Y)".

## What to publish, and what not to

* **Publish:** the code, `PREREGISTRATION.md`, `DEVIATIONS.md`, the summary JSON files, the figures, and the per-checkpoint scores.
* **Do not publish:** the decrypted Turner et al. datasets (the notebook keeps them in `/tmp` on Kaggle for this reason), or raw harmful generations. Offer the raw generations on request.

## How this was tested

Every code path was run end to end on CPU with tiny randomly initialised Qwen2 and OLMo2 models, a locally trained tokenizer and a mock judge. That covered the real clone and decryption of the Turner et al. datasets, both workers, resuming, the time budget and every plot. It was not run on a GPU or on the real models: the Kaggle dry run is the first real test, which is why it comes first.

## References

* Betley et al. (2025). Emergent Misalignment. arXiv:2502.17424.
* Turner, Soligo, Taylor, Rajamanoharan, Nanda (2025). Model Organisms for Emergent Misalignment. arXiv:2506.11613. Code and data: github.com/clarifying-EM/model-organisms-for-EM.
* Soligo, Turner, Rajamanoharan, Nanda (2025). Convergent Linear Representations of Emergent Misalignment. arXiv:2506.11618.
* OLMo-2 1B checkpoints: huggingface.co/allenai/OLMo-2-0425-1B.
