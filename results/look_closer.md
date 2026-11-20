## Look closer until confident: offline simulation on the label-reading data

### Does confidence separate right from wrong at a fixed distance?

AUROC of the reader's confidence for choosing the correct box, within one camera distance. `–` means every choice at that distance was correct (or every one wrong).

| model               | 1.0 m          | 0.7 m          | 0.5 m          | 0.35 m         | 0.22 m         |
|:--------------------|:---------------|:---------------|:---------------|:---------------|:---------------|
| glm-ocr             | 0.54 (acc 50%) | 0.83 (acc 77%) | 0.87 (acc 93%) | 0.96 (acc 99%) | –              |
| paddleocr-vl-1.6    | 0.55 (acc 64%) | 0.78 (acc 81%) | 0.75 (acc 97%) | –              | 0.80 (acc 98%) |
| qwen2.5-vl-3b       | 0.49 (acc 47%) | 0.85 (acc 80%) | 0.92 (acc 91%) | 0.97 (acc 99%) | –              |
| qwen3.5-2b          | 0.55 (acc 65%) | 0.81 (acc 81%) | 0.90 (acc 96%) | 1.00 (acc 99%) | –              |
| smolvlm2-500m       | 0.63 (acc 51%) | 0.82 (acc 63%) | 0.92 (acc 81%) | 0.94 (acc 86%) | 0.95 (acc 81%) |
| smolvlm2-500m-64tok | 0.46 (acc 47%) | 0.74 (acc 56%) | 0.85 (acc 78%) | 0.95 (acc 81%) | 0.89 (acc 82%) |

### Adaptive stopping against fixed distances

`moves` counts camera steps towards the label (0 = decided from 1.0 m, 4 = went to 0.22 m). Intervals are the 5th to 95th percentile over random half/half splits of the trajectories.

| model               | target error   |   trajectories | error at 1.0 m, no moves   | error at 0.22 m, 4 moves   | fixed distance meeting target   | adaptive: error among committed   | adaptive: splits over target   | adaptive: abstains   | adaptive: moves   | adaptive: decided at 1.0 m   | splits with no threshold   |
|:--------------------|:---------------|---------------:|:---------------------------|:---------------------------|:--------------------------------|:----------------------------------|:-------------------------------|:---------------------|:------------------|:-----------------------------|:---------------------------|
| glm-ocr             | 5%             |            108 | 50%                        | 0%                         | 0.35 m (3 moves)                | 5% [0–14]                         | 40%                            | 11% [6–17]           | 1.77 [1.48–1.98]  | 4%                           | 0%                         |
| glm-ocr             | 10%            |            108 | 50%                        | 0%                         | 0.5 m (2 moves)                 | 11% [2–21]                        | 47%                            | 7% [2–13]            | 1.54 [1.30–1.83]  | 9%                           | 0%                         |
| paddleocr-vl-1.6    | 5%             |            108 | 36%                        | 2%                         | 0.5 m (2 moves)                 | 5% [0–11]                         | 42%                            | 2% [0–7]             | 1.50 [1.17–2.11]  | 10%                          | 0%                         |
| paddleocr-vl-1.6    | 10%            |            108 | 36%                        | 2%                         | 0.5 m (2 moves)                 | 11% [2–20]                        | 47%                            | 0% [0–0]             | 1.16 [0.93–1.37]  | 20%                          | 0%                         |
| qwen2.5-vl-3b       | 5%             |            108 | 53%                        | 0%                         | 0.35 m (3 moves)                | 4% [0–13]                         | 45%                            | 0% [0–0]             | 1.62 [1.39–1.80]  | 1%                           | 0%                         |
| qwen2.5-vl-3b       | 10%            |            108 | 53%                        | 0%                         | 0.5 m (2 moves)                 | 11% [2–20]                        | 53%                            | 0% [0–0]             | 1.37 [1.17–1.56]  | 7%                           | 0%                         |
| qwen3.5-2b          | 5%             |            108 | 35%                        | 0%                         | 0.5 m (2 moves)                 | 5% [0–13]                         | 48%                            | 5% [0–13]            | 1.54 [1.15–2.11]  | 12%                          | 0%                         |
| qwen3.5-2b          | 10%            |            108 | 35%                        | 0%                         | 0.5 m (2 moves)                 | 10% [2–19]                        | 48%                            | 1% [0–4]             | 1.21 [0.98–1.41]  | 22%                          | 0%                         |
| smolvlm2-500m       | 5%             |            108 | 49%                        | 19%                        | none                            | 4% [0–12]                         | 42%                            | 30% [20–41]          | 2.56 [2.00–2.96]  | 1%                           | 0%                         |
| smolvlm2-500m       | 10%            |            108 | 49%                        | 19%                        | none                            | 9% [0–21]                         | 36%                            | 21% [11–31]          | 2.03 [1.59–2.52]  | 6%                           | 0%                         |
| smolvlm2-500m-64tok | 5%             |            108 | 53%                        | 18%                        | none                            | 5% [0–14]                         | 48%                            | 37% [28–44]          | 2.83 [2.50–3.06]  | 0%                           | 0%                         |
| smolvlm2-500m-64tok | 10%            |            108 | 53%                        | 18%                        | none                            | 10% [3–21]                        | 43%                            | 31% [20–41]          | 2.54 [2.17–2.91]  | 0%                           | 0%                         |
