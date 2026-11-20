# Pilot results

## Hypothesis 1 — fabrication, confidence and compression on handwriting

Tables that pool languages use only the lines a model read at both precisions, so the two precisions are compared on the same lines.

### Reading accuracy by legibility (a line is correct when its character error rate is at most 10%)

`h24`, `h14`, `h9` are the same lines reduced to a height of 24, 14 and 9 pixels.

| model            | precision   | data     | clean                                | h24                                  | h14                                  | h9                                   |
|:-----------------|:------------|:---------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|
| glm-ocr          | 4bit        | casia_zh | 75% correct; median CER 0.04 (n=120) | 55% correct; median CER 0.09 (n=120) | 12% correct; median CER 0.46 (n=120) | 0% correct; median CER 0.93 (n=120)  |
| glm-ocr          | 4bit        | iam_en   | 91% correct; median CER 0.00 (n=120) | 84% correct; median CER 0.00 (n=120) | 53% correct; median CER 0.08 (n=120) | 12% correct; median CER 0.40 (n=120) |
| glm-ocr          | 4bit        | uit_vi   | 0% correct; median CER 0.54 (n=120)  | 0% correct; median CER 0.97 (n=120)  | 0% correct; median CER 1.03 (n=120)  | 0% correct; median CER 1.05 (n=120)  |
| glm-ocr          | bf16        | casia_zh | 90% correct; median CER 0.00 (n=120) | 69% correct; median CER 0.04 (n=120) | 27% correct; median CER 0.27 (n=120) | 2% correct; median CER 0.79 (n=120)  |
| glm-ocr          | bf16        | iam_en   | 98% correct; median CER 0.00 (n=120) | 93% correct; median CER 0.00 (n=120) | 70% correct; median CER 0.02 (n=120) | 31% correct; median CER 0.26 (n=120) |
| glm-ocr          | bf16        | uit_vi   | 0% correct; median CER 0.44 (n=120)  | 0% correct; median CER 0.87 (n=120)  | 0% correct; median CER 1.18 (n=120)  | 0% correct; median CER 1.33 (n=120)  |
| paddleocr-vl-1.6 | 4bit        | casia_zh | 92% correct; median CER 0.00 (n=120) | 84% correct; median CER 0.00 (n=120) | 42% correct; median CER 0.15 (n=120) | 2% correct; median CER 0.78 (n=120)  |
| paddleocr-vl-1.6 | 4bit        | iam_en   | 84% correct; median CER 0.02 (n=120) | 72% correct; median CER 0.04 (n=120) | 45% correct; median CER 0.13 (n=120) | 9% correct; median CER 0.47 (n=120)  |
| paddleocr-vl-1.6 | 4bit        | uit_vi   | 8% correct; median CER 0.36 (n=37)   | 0% correct; median CER 0.91 (n=37)   | 0% correct; median CER 1.00 (n=37)   | 0% correct; median CER 1.00 (n=37)   |
| paddleocr-vl-1.6 | bf16        | casia_zh | 95% correct; median CER 0.00 (n=120) | 87% correct; median CER 0.00 (n=120) | 50% correct; median CER 0.10 (n=120) | 2% correct; median CER 0.77 (n=120)  |
| paddleocr-vl-1.6 | bf16        | iam_en   | 87% correct; median CER 0.01 (n=120) | 76% correct; median CER 0.02 (n=120) | 44% correct; median CER 0.13 (n=120) | 12% correct; median CER 0.41 (n=120) |
| paddleocr-vl-1.6 | bf16        | uit_vi   | 9% correct; median CER 0.30 (n=120)  | 1% correct; median CER 0.73 (n=120)  | 0% correct; median CER 0.95 (n=120)  | 0% correct; median CER 1.00 (n=120)  |
| qwen2.5-vl-3b    | 4bit        | casia_zh | 71% correct; median CER 0.04 (n=120) | 42% correct; median CER 0.12 (n=120) | 8% correct; median CER 0.65 (n=120)  | 1% correct; median CER 1.11 (n=120)  |
| qwen2.5-vl-3b    | 4bit        | iam_en   | 82% correct; median CER 0.00 (n=120) | 70% correct; median CER 0.04 (n=120) | 36% correct; median CER 0.17 (n=120) | 1% correct; median CER 0.90 (n=120)  |
| qwen2.5-vl-3b    | 4bit        | uit_vi   | 7% correct; median CER 0.37 (n=120)  | 1% correct; median CER 0.70 (n=120)  | 1% correct; median CER 0.91 (n=120)  | 0% correct; median CER 0.99 (n=120)  |
| qwen2.5-vl-3b    | bf16        | casia_zh | 82% correct; median CER 0.03 (n=120) | 64% correct; median CER 0.07 (n=120) | 11% correct; median CER 0.60 (n=120) | 1% correct; median CER 1.33 (n=120)  |
| qwen2.5-vl-3b    | bf16        | iam_en   | 85% correct; median CER 0.00 (n=120) | 70% correct; median CER 0.02 (n=120) | 42% correct; median CER 0.16 (n=120) | 2% correct; median CER 0.80 (n=120)  |
| qwen2.5-vl-3b    | bf16        | uit_vi   | 11% correct; median CER 0.31 (n=120) | 2% correct; median CER 0.64 (n=120)  | 1% correct; median CER 0.89 (n=120)  | 0% correct; median CER 0.96 (n=120)  |
| qwen3.5-2b       | 4bit        | casia_zh | 79% correct; median CER 0.03 (n=120) | 59% correct; median CER 0.06 (n=120) | 18% correct; median CER 0.33 (n=120) | 2% correct; median CER 0.92 (n=120)  |
| qwen3.5-2b       | 4bit        | iam_en   | 88% correct; median CER 0.00 (n=120) | 75% correct; median CER 0.00 (n=120) | 57% correct; median CER 0.07 (n=120) | 18% correct; median CER 0.37 (n=120) |
| qwen3.5-2b       | 4bit        | uit_vi   | 12% correct; median CER 0.35 (n=120) | 2% correct; median CER 0.77 (n=120)  | 0% correct; median CER 0.86 (n=120)  | 0% correct; median CER 0.89 (n=120)  |
| qwen3.5-2b       | bf16        | casia_zh | 85% correct; median CER 0.03 (n=120) | 73% correct; median CER 0.04 (n=120) | 23% correct; median CER 0.26 (n=120) | 2% correct; median CER 0.86 (n=120)  |
| qwen3.5-2b       | bf16        | iam_en   | 86% correct; median CER 0.00 (n=120) | 78% correct; median CER 0.00 (n=120) | 60% correct; median CER 0.06 (n=120) | 18% correct; median CER 0.30 (n=120) |
| qwen3.5-2b       | bf16        | uit_vi   | 15% correct; median CER 0.24 (n=120) | 4% correct; median CER 0.71 (n=120)  | 0% correct; median CER 0.84 (n=120)  | 0% correct; median CER 0.90 (n=120)  |

### Fabrication: output when nothing is legible

`blur` and `blank` contain no readable text. `returns text` is an answer with at least 3 letters, digits or CJK characters. `invented text` leaves out answers that only repeat the instruction ("The text in the image is: ..."). `h9 wrong but fluent` is the share of 9-pixel lines that were read wrongly yet answered with at least half as many characters as the true text.

| model            | precision   | blur: returns text   | blur: invented text   | blank: returns text   | blank: invented text   | h9 wrong but fluent   |
|:-----------------|:------------|:---------------------|:----------------------|:----------------------|:-----------------------|:----------------------|
| glm-ocr          | 4bit        | 0% [0–1] (n=360)     | 0% [0–1] (n=360)      | 0% [0–1] (n=360)      | 0% [0–1] (n=360)       | 89% [85–92] (n=360)   |
| glm-ocr          | bf16        | 0% [0–1] (n=360)     | 0% [0–1] (n=360)      | 0% [0–1] (n=360)      | 0% [0–1] (n=360)       | 83% [79–87] (n=360)   |
| paddleocr-vl-1.6 | 4bit        | 0% [0–1] (n=277)     | 0% [0–1] (n=277)      | 0% [0–1] (n=277)      | 0% [0–1] (n=277)       | 88% [84–92] (n=277)   |
| paddleocr-vl-1.6 | bf16        | 5% [3–9] (n=277)     | 5% [3–9] (n=277)      | 0% [0–2] (n=277)      | 0% [0–2] (n=277)       | 83% [78–87] (n=277)   |
| qwen2.5-vl-3b    | 4bit        | 52% [47–57] (n=360)  | 5% [3–7] (n=360)      | 50% [45–55] (n=360)   | 0% [0–2] (n=360)       | 79% [75–83] (n=360)   |
| qwen2.5-vl-3b    | bf16        | 92% [89–95] (n=360)  | 11% [8–14] (n=360)    | 91% [88–94] (n=360)   | 9% [6–12] (n=360)      | 87% [83–90] (n=360)   |
| qwen3.5-2b       | 4bit        | 0% [0–1] (n=360)     | 0% [0–1] (n=360)      | 0% [0–1] (n=360)      | 0% [0–1] (n=360)       | 86% [82–89] (n=360)   |
| qwen3.5-2b       | bf16        | 7% [5–10] (n=360)    | 7% [5–10] (n=360)     | 0% [0–1] (n=360)      | 0% [0–1] (n=360)       | 87% [83–90] (n=360)   |

### Repetition loops

A loop is an answer of at least 8 space-separated tokens with very few distinct ones (a crude detector that misses loops in unspaced Chinese). `as confident as a correct line` counts loops whose mean token log-probability is at least the median of that model's correctly read lines.

| model            | precision   | loops             | as confident as a correct line   |
|:-----------------|:------------|:------------------|:---------------------------------|
| glm-ocr          | 4bit        | 6% [5–8] (n=1440) | 2 of 90                          |
| glm-ocr          | bf16        | 4% [3–5] (n=1440) | 0 of 62                          |
| paddleocr-vl-1.6 | 4bit        | 1% [1–2] (n=1108) | 0 of 14                          |
| paddleocr-vl-1.6 | bf16        | 0% [0–1] (n=1108) | 0 of 1                           |
| qwen2.5-vl-3b    | 4bit        | 0% [0–1] (n=1440) | 0 of 6                           |
| qwen2.5-vl-3b    | bf16        | 1% [0–1] (n=1440) | 0 of 8                           |
| qwen3.5-2b       | 4bit        | 4% [3–5] (n=1440) | 3 of 55                          |
| qwen3.5-2b       | bf16        | 2% [2–3] (n=1440) | 2 of 35                          |

### Does confidence separate right from wrong?

AUROC of mean token log-probability over all readable lines (clean and reduced): 0.5 is useless, 1.0 is perfect. `selective` picks a threshold on half the lines for a 10% or 20% error target and reports the error and the share of lines still answered on the other half.

| model            | precision   |   lines | accuracy answering everything   |   AUROC right vs wrong | selective @10%                 | selective @20%                 |
|:-----------------|:------------|--------:|:--------------------------------|-----------------------:|:-------------------------------|:-------------------------------|
| glm-ocr          | 4bit        |    1440 | 32%                             |                   0.95 | error 9%, answers 19% (n=140)  | error 18%, answers 32% (n=228) |
| glm-ocr          | bf16        |    1440 | 40%                             |                   0.97 | error 11%, answers 38% (n=275) | error 19%, answers 48% (n=342) |
| paddleocr-vl-1.6 | 4bit        |    1108 | 47%                             |                   0.95 | error 9%, answers 43% (n=239)  | error 22%, answers 55% (n=306) |
| paddleocr-vl-1.6 | bf16        |    1108 | 49%                             |                   0.97 | error 12%, answers 51% (n=285) | error 21%, answers 62% (n=342) |
| qwen2.5-vl-3b    | 4bit        |    1440 | 27%                             |                   0.97 | error 7%, answers 20% (n=147)  | error 15%, answers 29% (n=206) |
| qwen2.5-vl-3b    | bf16        |    1440 | 31%                             |                   0.97 | error 8%, answers 29% (n=208)  | error 17%, answers 36% (n=259) |
| qwen3.5-2b       | 4bit        |    1440 | 34%                             |                   0.94 | error 5%, answers 20% (n=143)  | error 14%, answers 30% (n=215) |
| qwen3.5-2b       | bf16        |    1440 | 37%                             |                   0.96 | error 9%, answers 32% (n=229)  | error 22%, answers 44% (n=316) |

### Confidence within one legibility level

The pooled AUROC above partly reflects that small lines are both harder and less confident. This is the AUROC among lines of one language at one size, with the share read correctly.

| model            | precision   | data     | clean              | h24                | h14                | h9                 |
|:-----------------|:------------|:---------|:-------------------|:-------------------|:-------------------|:-------------------|
| glm-ocr          | 4bit        | casia_zh | 0.92 (75% correct) | 0.95 (55% correct) | 0.94 (12% correct) | –                  |
| glm-ocr          | 4bit        | iam_en   | 0.80 (91% correct) | 0.89 (84% correct) | 0.92 (53% correct) | 0.88 (12% correct) |
| glm-ocr          | 4bit        | uit_vi   | –                  | –                  | –                  | –                  |
| glm-ocr          | bf16        | casia_zh | 0.93 (90% correct) | 0.95 (69% correct) | 0.97 (27% correct) | 0.95 (2% correct)  |
| glm-ocr          | bf16        | iam_en   | 0.50 (98% correct) | 0.83 (93% correct) | 0.92 (70% correct) | 0.94 (31% correct) |
| glm-ocr          | bf16        | uit_vi   | –                  | –                  | –                  | –                  |
| paddleocr-vl-1.6 | 4bit        | casia_zh | 1.00 (92% correct) | 0.96 (84% correct) | 0.96 (42% correct) | 0.97 (2% correct)  |
| paddleocr-vl-1.6 | 4bit        | iam_en   | 0.85 (84% correct) | 0.91 (72% correct) | 0.91 (45% correct) | 0.94 (9% correct)  |
| paddleocr-vl-1.6 | 4bit        | uit_vi   | 0.99 (8% correct)  | –                  | –                  | –                  |
| paddleocr-vl-1.6 | bf16        | casia_zh | 0.97 (95% correct) | 0.98 (87% correct) | 0.96 (50% correct) | 0.98 (2% correct)  |
| paddleocr-vl-1.6 | bf16        | iam_en   | 0.88 (87% correct) | 0.92 (76% correct) | 0.93 (44% correct) | 0.96 (12% correct) |
| paddleocr-vl-1.6 | bf16        | uit_vi   | 0.96 (9% correct)  | 0.99 (1% correct)  | –                  | –                  |
| qwen2.5-vl-3b    | 4bit        | casia_zh | 0.89 (71% correct) | 0.92 (42% correct) | 0.98 (8% correct)  | 1.00 (1% correct)  |
| qwen2.5-vl-3b    | 4bit        | iam_en   | 0.83 (82% correct) | 0.91 (70% correct) | 0.91 (36% correct) | 0.98 (1% correct)  |
| qwen2.5-vl-3b    | 4bit        | uit_vi   | 0.95 (7% correct)  | 0.97 (1% correct)  | 0.80 (1% correct)  | –                  |
| qwen2.5-vl-3b    | bf16        | casia_zh | 0.89 (82% correct) | 0.90 (64% correct) | 0.97 (11% correct) | 1.00 (1% correct)  |
| qwen2.5-vl-3b    | bf16        | iam_en   | 0.85 (85% correct) | 0.91 (70% correct) | 0.96 (42% correct) | 0.90 (2% correct)  |
| qwen2.5-vl-3b    | bf16        | uit_vi   | 0.91 (11% correct) | 0.90 (2% correct)  | 0.97 (1% correct)  | –                  |
| qwen3.5-2b       | 4bit        | casia_zh | 0.87 (79% correct) | 0.94 (59% correct) | 0.93 (18% correct) | 0.84 (2% correct)  |
| qwen3.5-2b       | 4bit        | iam_en   | 0.92 (88% correct) | 0.92 (75% correct) | 0.93 (57% correct) | 0.95 (18% correct) |
| qwen3.5-2b       | 4bit        | uit_vi   | 0.87 (12% correct) | 0.85 (2% correct)  | –                  | –                  |
| qwen3.5-2b       | bf16        | casia_zh | 0.90 (85% correct) | 0.94 (73% correct) | 0.96 (23% correct) | 0.91 (2% correct)  |
| qwen3.5-2b       | bf16        | iam_en   | 0.90 (86% correct) | 0.91 (78% correct) | 0.94 (60% correct) | 0.97 (18% correct) |
| qwen3.5-2b       | bf16        | uit_vi   | 0.89 (15% correct) | 0.93 (4% correct)  | –                  | –                  |

### Same, per language

| model            | precision   | data     | accuracy   |   AUROC | selective @10%                 |
|:-----------------|:------------|:---------|:-----------|--------:|:-------------------------------|
| glm-ocr          | 4bit        | casia_zh | 36%        |    0.94 | error 16%, answers 20% (n=49)  |
| glm-ocr          | 4bit        | iam_en   | 60%        |    0.94 | error 10%, answers 60% (n=143) |
| glm-ocr          | 4bit        | uit_vi   | 0%         |  nan    | no threshold found             |
| glm-ocr          | bf16        | casia_zh | 47%        |    0.98 | error 9%, answers 47% (n=113)  |
| glm-ocr          | bf16        | iam_en   | 73%        |    0.94 | error 17%, answers 83% (n=200) |
| glm-ocr          | bf16        | uit_vi   | 0%         |  nan    | no threshold found             |
| paddleocr-vl-1.6 | 4bit        | casia_zh | 55%        |    0.99 | error 12%, answers 57% (n=137) |
| paddleocr-vl-1.6 | 4bit        | iam_en   | 53%        |    0.93 | error 18%, answers 53% (n=128) |
| paddleocr-vl-1.6 | 4bit        | uit_vi   | 2%         |    0.8  | no threshold found             |
| paddleocr-vl-1.6 | bf16        | casia_zh | 59%        |    0.99 | error 7%, answers 59% (n=141)  |
| paddleocr-vl-1.6 | bf16        | iam_en   | 55%        |    0.95 | error 14%, answers 57% (n=138) |
| paddleocr-vl-1.6 | bf16        | uit_vi   | 2%         |    0.92 | no threshold found             |
| qwen2.5-vl-3b    | 4bit        | casia_zh | 31%        |    0.97 | error 19%, answers 28% (n=67)  |
| qwen2.5-vl-3b    | 4bit        | iam_en   | 47%        |    0.95 | error 26%, answers 57% (n=136) |
| qwen2.5-vl-3b    | 4bit        | uit_vi   | 2%         |    0.94 | no threshold found             |
| qwen2.5-vl-3b    | bf16        | casia_zh | 39%        |    0.97 | error 22%, answers 43% (n=104) |
| qwen2.5-vl-3b    | bf16        | iam_en   | 50%        |    0.96 | error 22%, answers 57% (n=138) |
| qwen2.5-vl-3b    | bf16        | uit_vi   | 3%         |    0.93 | no threshold found             |
| qwen3.5-2b       | 4bit        | casia_zh | 39%        |    0.93 | error 6%, answers 22% (n=53)   |
| qwen3.5-2b       | 4bit        | iam_en   | 59%        |    0.95 | error 10%, answers 60% (n=143) |
| qwen3.5-2b       | 4bit        | uit_vi   | 4%         |    0.83 | no threshold found             |
| qwen3.5-2b       | bf16        | casia_zh | 46%        |    0.97 | error 3%, answers 37% (n=89)   |
| qwen3.5-2b       | bf16        | iam_en   | 60%        |    0.95 | error 12%, answers 65% (n=156) |
| qwen3.5-2b       | bf16        | uit_vi   | 5%         |    0.85 | no threshold found             |

### Second Vietnamese source: scanned UIT-HWDB lines, full precision

A check that the Vietnamese result is not specific to the pen-stroke renderings of VNOnDB.

| model            | clean                                | h24                                  | h14                                 | h9                                  |   AUROC right vs wrong | selective @10%     | blank: returns text   |
|:-----------------|:-------------------------------------|:-------------------------------------|:------------------------------------|:------------------------------------|-----------------------:|:-------------------|:----------------------|
| glm-ocr          | 1% correct; median CER 0.28 (n=120)  | 1% correct; median CER 0.45 (n=120)  | 0% correct; median CER 0.91 (n=120) | 0% correct; median CER 0.95 (n=120) |                   0.73 | no threshold found | 0% [0–3] (n=120)      |
| paddleocr-vl-1.6 | 17% correct; median CER 0.21 (n=120) | 3% correct; median CER 0.37 (n=120)  | 0% correct; median CER 0.70 (n=120) | 0% correct; median CER 0.92 (n=120) |                   0.93 | no threshold found | 0% [0–3] (n=120)      |
| qwen2.5-vl-3b    | 55% correct; median CER 0.09 (n=120) | 12% correct; median CER 0.38 (n=120) | 0% correct; median CER 0.92 (n=120) | 0% correct; median CER 1.00 (n=120) |                   0.96 | no threshold found | 100% [97–100] (n=120) |
| qwen3.5-2b       | 46% correct; median CER 0.12 (n=120) | 12% correct; median CER 0.35 (n=120) | 0% correct; median CER 0.82 (n=120) | 0% correct; median CER 1.00 (n=120) |                   0.9  | no threshold found | 2% [0–6] (n=120)      |
