# Contributing

The most useful contribution is a result for a model that is not on the leaderboard yet. It takes a few minutes on one GPU.

## Add a model to the leaderboard

1. Run the full chart (not `--quick`):

   ```bash
   python canitread.py --model <hugging-face-repo-id> --save-answers
   ```

2. Check the printed table looks sane. If the model needs a fixed task prompt (most OCR models do), pass it with `--kind ocr --prompt "<prompt>"` and say which prompt you used in the pull request.
3. Open a pull request that adds the two files written to `results/canitread/` and nothing else. The leaderboard in the README is regenerated from those files with `python canitread.py --leaderboard`.

Rules that keep the table comparable:

- Use the chart as shipped. Do not change `SIZES`, the pair list or the label drawing in the same pull request as a result.
- Greedy decoding only, which is the default.
- Report quantised runs as their own row (`--precision 4bit` does this for you).
- Results from an API-only model are welcome if you describe how each label image was sent and what was returned.

## Report a model that will not load

Open an issue with the repository id, the command you ran and the last lines of the error. Models that need `trust_remote_code` are not supported yet.

## Other things worth a pull request

- **More scripts and languages.** Look-alike pairs live in `labels.py`. New pairs should be real confusable strings, with a short note on where the confusion comes from.
- **Harder charts.** Rotation, glare, handwriting-style fonts and curved surfaces, as options that leave the default chart unchanged.
- **A closed-loop robot test.** Running an open policy on the pick-by-label scenes is the first item on the roadmap.

## Changing the chart

Any change to what a model is shown changes every score. Such a change bumps `CHART_VERSION` in `canitread.py`, and results from different chart versions are never mixed in one table.
