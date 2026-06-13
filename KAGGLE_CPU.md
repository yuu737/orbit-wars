# Kaggle CPU Evaluation Setup

This repository can be cloned directly inside a Kaggle Notebook and used for CPU-based local evaluations.

## 1. Create a Kaggle Notebook

- Open a new Kaggle Notebook.
- Turn `Internet` on in the notebook settings so `git clone` works.
- CPU is enough for the current evaluation flow.

## 2. Clone the repository

```python
!git clone https://github.com/yuu737/orbit-wars.git
%cd /kaggle/working/orbit-wars
```

## 3. Install dependencies

```python
!python -m pip install -q -r requirements.txt
```

If the preinstalled PyTorch is already available, this usually finishes quickly.

## 4. Run an evaluation

Example: compare the current opening branch against `hairate2`.

```python
!python evaluate.py --players 2 --agent bots/hairate9_opening.py --opponent bots/hairate2.py --games 40 --both-seats --workers 4 --seed-start 90000000
```

Example: compare the Kaggle-stable target-retake branch against `hairate5`.

```python
!python evaluate.py --players 2 --agent bots/hairate8_target_w20.py --opponent bots/hairate5.py --games 40 --both-seats --workers 4 --seed-start 90000000
```

## 5. Suggested Kaggle workflow

- Use `workers=2` to `4` first to avoid noisy notebook failures.
- Save summaries to text files when running many experiments.
- Keep one baseline fixed, for example `bots/hairate5.py` or `bots/hairate8_target_w20.py`.
- Separate tuning seeds from validation seeds.

## 6. Current useful agent files

- `bots/hairate5.py`: strong classic baseline.
- `bots/hairate8_target_w20.py`: target-retake weighting branch that scored best on your recent public submissions.
- `bots/hairate9_opening.py`: experimental opening-aware branch.

## 7. Submitting to the competition

For Kaggle competition submission, keep using a single submission file rooted at `main.py` or a zip/tar bundle with `main.py` at the top level. This repository is mainly organized for experimentation, so choose the agent file you want to submit and package that separately.
