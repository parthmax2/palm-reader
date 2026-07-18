# Palmistry AI

A backend API that reads a palm photo, detects the principal lines with computer vision,
matches them against a Hindu palmistry (**Samudrika Shastra / Hast Rekha Shastra**) knowledge
base, and generates a personalized reading in **English and Hindi** using **Gemini 2.5 Flash**.

> Readings are a **traditional interpretation for guidance and entertainment** — not medical,
> financial, legal, or predictive advice. Every response includes a disclaimer.

## How it works

```
palm photo
  └─ CV pipeline (app/cv)          MediaPipe warp → U-Net line segmentation → K-means classify
       └─ FeatureVector            heart / head / life: present, length, curvature, confidence
            └─ rule engine (app/knowledge)   match Samudrika Shastra rules (EN + HI meanings)
                 └─ generation (app/generation)   Gemini 2.5 Flash → bilingual reading (JSON)
                      └─ ReadingResponse
```

The CV model is the **U-Net Context Fusion** architecture from
[yeonsumia/palmistry](https://github.com/yeonsumia/palmistry) (Apache-2.0), vendored under
`app/cv/vendor/` with its pretrained checkpoint. It detects the **Heart (Hridaya), Head
(Mastishka), and Life (Jeevan)** lines.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# then:
pip install -r requirements.txt
cp .env.example .env      # add your GEMINI_API_KEY (or set it in the environment)
```

> **Python version:** use **3.10** — `mediapipe` wheels are reliable there. (3.13 currently
> ships a broken mediapipe wheel missing the `solutions` API.)

If `GEMINI_API_KEY` is not set, the API still works and falls back to a deterministic
template reading built directly from the matched rule meanings (`generation: "template-fallback"`).

## Run

```bash
uvicorn app.main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health: `GET /v1/health`

## Endpoints (`/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Status + whether Gemini is enabled |
| POST | `/features` | Image → feature vector (CV only, no LLM) |
| POST | `/readings` | Image → full bilingual reading |
| POST | `/readings/from-features` | Feature vector → reading (skip CV; great for testing) |

### Example: full reading from an image

```bash
curl -X POST http://127.0.0.1:8000/v1/readings \
  -F "image=@samples/hand1.jpg" \
  -F 'options={"languages":["en","hi"],"detail":"standard"}'
```

### Example: reading from a manual feature vector (no image)

```bash
curl -X POST http://127.0.0.1:8000/v1/readings/from-features \
  -H "Content-Type: application/json" \
  -d '{"features":{"hand":"right",
        "heart":{"present":true,"length":"long","curved":true},
        "head":{"present":true,"length":"short","curved":false},
        "life":{"present":true,"length":"long","curved":true}},
       "options":{"languages":["en","hi"],"detail":"standard"}}'
```

## Project layout

```
app/
  main.py            FastAPI app
  config.py          settings (.env: GEMINI_API_KEY, model, temperature, retention)
  service.py         orchestration: CV → rules → generation
  api/routes.py      endpoints
  models/schemas.py  FeatureVector, Reading, request/response models
  cv/
    pipeline.py      image → FeatureVector (uses the vendored model)
    vendor/          vendored palmistry CV code (Apache-2.0)
    checkpoint/      pretrained U-Net weights (.pth)
  knowledge/
    rules.json       Samudrika Shastra rulebook (EN + HI, versioned)
    engine.py        rule matching
  generation/
    prompts.py       system instruction + user prompt + JSON schema (guardrails)
    reader.py        Gemini 2.5 Flash call + template fallback
samples/             example palm images
ARCHITECTURE.md      full design document
```

## Feature status (CV pipeline)

| Feature | Status | Notes |
|---|---|---|
| Heart / Head / Life lines | ✅ working | length, curvature, confidence |
| **Fork detection** (`fork_end`) | ✅ working | from skeleton junctions; verified True at a junction, False in open space |
| **Fate line** (Bhagya Rekha) | ⚠️ infra ready, model-limited | selector + schema + rules are in place, but the pretrained model does **not** segment a 4th line on the sample set (only 3 candidates are produced). `fate` will populate once the model is fine-tuned to include it, or a dedicated fate detector is added. |
| **Breaks / islands** | ⚠️ schema + rules ready, detection dormant | rules keyed on `breaks_present` auto-activate once detection lands; not populated now to avoid false positives on fragmented output |
| Mounts (Parvat), hand shape | ⬜ not started | specified in `ARCHITECTURE.md` |

## Known limitations & roadmap

- **Line completeness (the main accuracy gap):** the model reliably *locates* the 3 major
  lines but captures mainly their central segment, so lines are often classed "short."
  I tested morphological **fragment-bridging** (dilate → skeletonize) to lengthen them — it
  produced duplicate/parallel skeleton branches and did **not** reliably help, so it is off
  by default (`_BRIDGE = 0` in `app/cv/pipeline.py`, kept configurable). **The real fix is
  fine-tuning / retraining the segmentation model** (see `ARCHITECTURE.md` §2.4, §7).
- **Fate line needs model work:** recovering it from the current model's candidates is not
  possible because it isn't segmented. Options: fine-tune on a dataset that labels the fate
  line, or add a dedicated classical detector for the vertical central crease.
- **Rulebook:** 29 starter rules across personality, relationships, career, intellect, health —
  covering length, curvature, forks, breaks (dormant), and combinations. Grow with a domain expert.
- **Privacy:** palm images are biometric-adjacent PII. `DELETE_IMAGES_AFTER_PROCESSING=true`
  (default) removes uploads immediately after processing.
- **Scale:** add auth, rate limiting, and an async job queue before production (see ARCHITECTURE.md §5).
```
