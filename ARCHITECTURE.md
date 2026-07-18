# Palm Reader AI — Architecture Plan

**Product:** A backend/API service that accepts a palm photo, detects the hand's lines, mounts, and shape using computer vision, matches them against a structured Hindu palmistry (Samudrika Shastra / Hast Rekha Shastra) knowledge base, and generates a personalized reading in **English and Hindi** using **Gemini 2.5 Flash**.

**Delivery target:** Backend / REST API only (no UI in v1).

---

## 0. Guiding principles & honest framing

Palmistry is a **traditional belief system**, not a scientifically validated predictor. The product must reflect this:

- Readings are framed as **traditional interpretation / guidance / entertainment**, never as guaranteed facts, and never as **medical, financial, legal, or lifespan** predictions.
- Every API response includes a `disclaimer` field.
- This is both an ethical stance and a practical one (app-store and payment-processor policies reject "guaranteed prediction" apps).

The impressive, *real* AI is the CV pipeline (detecting actual palm features from a photo) + grounded LLM narration. The LLM **narrates rules that matched detected features** — it does not invent predictions.

---

## 1. System overview

```
                        ┌─────────────────────────────────────────────┐
   POST /v1/readings    │                 API Layer                   │
   (image + options) ──▶│         (FastAPI, async, auth, rate-limit)  │
                        └───────────────┬─────────────────────────────┘
                                        │  job
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                       Processing Pipeline                      │
        │                                                                │
        │  1. Image intake & QA   ──▶  reject bad photos early           │
        │  2. Hand detection      ──▶  MediaPipe Hands (21 landmarks)    │
        │  3. Palm ROI + normalize ─▶  crop, deskew, standardize         │
        │  4. Line extraction     ──▶  CV / U-Net segmentation           │
        │  5. Mount analysis      ──▶  region depth/shading + landmarks  │
        │  6. Feature vector      ──▶  structured JSON of all findings   │
        │  7. Rule matching       ──▶  Samudrika Shastra knowledge base  │
        │  8. Reading generation  ──▶  Gemini 2.5 Flash (EN + HI)        │
        │  9. Annotated image     ──▶  overlay detected lines/mounts     │
        └───────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          Reading + features + annotated image
```

Two execution modes:
- **Synchronous** (`POST /v1/readings`) for a single quick reading (~3–8s).
- **Async job** (`POST /v1/jobs` → poll `GET /v1/jobs/{id}`) for heavier processing / batch.

---

## 2. Layer 1 — Computer Vision pipeline

This is the make-or-break layer. Build it as a chain of independently testable stages.

### 2.1 Image intake & quality assurance
Reject early with actionable errors so users get better retakes:
- Resolution ≥ threshold (e.g. min 1000px on long edge).
- Blur detection (variance of Laplacian below threshold → reject).
- Exposure check (over/under-exposed histogram → reject).
- Exactly one hand present, palm facing camera.

Returns structured errors like `{"code": "IMAGE_TOO_BLURRY", "hint": "Hold steady, ensure good lighting"}`.

### 2.2 Hand detection & landmarks — **MediaPipe Hands**
- Off-the-shelf, free, accurate. Produces **21 hand landmarks** (finger tips, joints, wrist).
- Gives us: handedness (left/right), finger positions, palm base — the coordinate system everything else is built on.

### 2.3 Palm ROI extraction & normalization
- Use landmarks to crop the palm region and **deskew/rotate** to a canonical orientation.
- Normalize scale and lighting (CLAHE contrast equalization).
- Output: a standardized palm image where feature positions are comparable across users.

### 2.4 Line extraction (the hard part)
Detect the four major lines and note secondary lines.

| Line (Sanskrit) | English | Region heuristic (from landmarks) |
|---|---|---|
| Hridaya Rekha | Heart line | Upper horizontal, below the fingers |
| Mastishka Rekha | Head line | Middle horizontal, across the palm |
| Jeevan Rekha | Life line | Curves around the thumb (Venus mount) |
| Bhagya Rekha | Fate line | Vertical, wrist → toward middle finger |
| (secondary) Surya, Budh, Vivah, etc. | Sun, Mercury, Marriage lines | Position-dependent |

**Three-tier approach (revised after pretrained-model research):**
1. **Fastest / recommended start — use an existing pretrained palmistry model.** [`yeonsumia/palmistry`](https://github.com/yeonsumia/palmistry) (Apache-2.0) **ships a committed 55.5 MB PyTorch checkpoint** (`checkpoint_aug_epoch70.pth`) purpose-built for palmistry: MediaPipe rectification → segmentation model → K-means line classification, with `read_palm.py` inference. This can skip dataset labeling and training entirely for the prototype and likely v1. **Validate on real-world photos before trusting it** (small research project, unknown training-set robustness).
2. **Prototype fallback:** Classical CV — grayscale → CLAHE → adaptive threshold → morphological thinning → contour detection → assign contours to lines by landmark-relative position. Fast, brittle to lighting.
3. **Production (only if 1 is insufficient):** Retrain a **segmentation model** — the **U-Net Context Fusion** architecture ([arXiv:2102.12127](https://arxiv.org/abs/2102.12127), reports F1 ≈ 99.42% on palm *lines*; paper only, no public weights) or fine-tune the yeonsumia weights. Best on the **CAS_Palm** labeled principal-line dataset ([PMC9590507](https://pmc.ncbi.nlm.nih.gov/articles/PMC9590507/), 5,251 labeled images, access on request), which largely removes the labeling bottleneck.

> **Critical distinction:** most "palmprint" pretrained models target **biometric recognition (identity)** or **palm-region ROI**, NOT palmistry line extraction. Only principal/palm-**line** models (above) detect the heart/head/life/fate lines you need.

### 2.5 Line attribute analysis
For each detected line, extract classical attributes (each maps to meanings in the rulebook):
- **Length** (relative to palm width), **depth/clarity**, **breaks**, **islands**, **chains**, **forks/branches**, **origin & termination point**, **curvature**.

### 2.6 Mount (Parvat) analysis
Seven classical mounts, located via landmarks, assessed for prominence (shading/relative elevation as a proxy for the raised flesh):

| Mount | Deity/Planet | Location |
|---|---|---|
| Guru Parvat | Jupiter | Base of index finger |
| Shani Parvat | Saturn | Base of middle finger |
| Surya Parvat | Sun | Base of ring finger |
| Budh Parvat | Mercury | Base of little finger |
| Shukra Parvat | Venus | Base of thumb (large area) |
| Chandra Parvat | Moon | Outer edge, opposite thumb |
| Mangal Parvat | Mars | Two zones (upper & lower Mars) |

### 2.7 Hand shape & finger analysis
- **Elemental hand type** (Earth/Prithvi, Fire/Agni, Air/Vayu, Water/Jal) from palm-to-finger aspect ratios.
- Finger length/shape, thumb angle/flexibility (from landmarks).

### 2.8 Output: the Feature Vector
A single structured JSON object — the **contract between CV and the knowledge base**. Example:

```json
{
  "hand": "right",
  "hand_shape": "fire",
  "confidence": 0.82,
  "lines": {
    "heart":  { "present": true, "length": "long", "depth": "deep", "breaks": 0, "fork_end": true, "origin": "under_jupiter" },
    "head":   { "present": true, "length": "medium", "attached_to_life": true, "fork_end": true },
    "life":   { "present": true, "length": "long", "depth": "deep", "breaks": 0, "curve": "wide" },
    "fate":   { "present": true, "origin": "wrist", "termination": "saturn", "breaks": 1 }
  },
  "mounts": {
    "jupiter": "prominent", "saturn": "normal", "sun": "flat",
    "mercury": "normal", "venus": "prominent", "moon": "normal", "mars": "normal"
  },
  "quality_flags": []
}
```

Every field carries a **per-feature confidence**; low-confidence features are narrated more tentatively (or omitted).

---

## 3. Layer 2 — Knowledge base (Samudrika Shastra rulebook)

**Data, not code.** Stored in a database (Postgres/SQLite) or versioned JSON so it can be expanded and reviewed by domain experts without touching the pipeline.

### 3.1 Rule schema
```json
{
  "id": "heart_long_deep_forked",
  "domain": "relationships",
  "conditions": {
    "lines.heart.length": "long",
    "lines.heart.depth": "deep",
    "lines.heart.fork_end": true
  },
  "weight": 0.9,
  "source": "Samudrika Shastra — Hridaya Rekha",
  "meaning_en": "A long, deep, forked heart line indicates a warm, balanced emotional nature and capacity for lasting, considerate relationships.",
  "meaning_hi": "लंबी, गहरी और अंत में द्विशाखा हृदय रेखा एक गर्म, संतुलित भावनात्मक स्वभाव और स्थायी, विचारशील संबंधों की क्षमता दर्शाती है।",
  "caution": "not_deterministic"
}
```

### 3.2 Rule engine
- Loads all rules, evaluates `conditions` against the feature vector, collects **matched rules**.
- Resolves conflicts (contradictory matches) by `weight` and confidence.
- Groups matches into **life domains**: personality, relationships/marriage (Vivah), career/fortune (Bhagya), health/vitality, intellect, wealth.
- Output: a set of matched, source-attributed interpretation fragments — **not free text yet**.

### 3.3 Content sourcing
- Ground rules in classical/traditional sources: **Samudrika Shastra**, Hast Rekha Shastra texts, widely-documented palmistry conventions.
- **Avoid harmful myths** — most importantly, the Life line does **not** indicate lifespan; encode vitality/resilience instead.
- Honor tradition's **hand conventions**: dominant hand = present/inclinations, non-dominant = inherited potential.
- Start with ~50–80 core rules for the prototype; grow to several hundred with expert review.

---

## 4. Layer 3 — Reading generation (Gemini 2.5 Flash)

The LLM turns matched rules into a warm, flowing, personalized reading — in **English and Hindi**. Model: **`gemini-2.5-flash`** via the Google Gen AI API (Google AI Studio key, or Vertex AI for GCP deployments).

### 4.1 Approach
- **Grounded generation:** the prompt contains ONLY the matched rule fragments + feature summary. The model is instructed to narrate *these* and not invent new predictions.
- **Structured output:** use Gemini's **`response_mime_type: application/json` + `response_schema`** (structured output) to force sections — Overview, Personality, Relationships, Career & Fortune, Health & Vitality, Notable Signs — as reliable JSON. This removes brittle text parsing.
- **Bilingual:** generate English and Hindi. Two clean options:
  1. One call returning both languages (schema with `en` / `hi` objects).
  2. Generate English, then a second call to translate/localize to natural Hindi (often higher Hindi quality). Recommended for quality; Flash is cheap/fast enough that the extra call is negligible.
- **Tone control:** a **system instruction** sets an empathetic, respectful, traditional palm-reader voice; forbids deterministic/medical claims; keeps the disclaimer intact.

### 4.2 Model choice & config
- **`gemini-2.5-flash`** — fast, low-cost, strong multilingual (good native Hindi). Well-suited to high request volume.
- Set a low-to-moderate `temperature` (~0.6) for warm but consistent readings.
- Gemini 2.5 Flash is a **thinking model**: keep a modest thinking budget for reasoning over rule conflicts, or set `thinking_budget: 0` to minimize latency/cost when readings are simple. Make it configurable.
- Optionally allow `gemini-2.5-pro` as a premium tier for more nuanced long-form readings — keep the model id configurable per request.
- Use **context caching** for the large static system instruction + rulebook context to cut cost/latency at scale.

### 4.3 Guardrails
- Configure Gemini **safety settings** appropriately, and rely primarily on the grounded prompt + schema.
- Post-generation validation: reject/regenerate if output contains banned claim patterns (lifespan, disease diagnosis, guaranteed money/marriage dates).
- Always append the `disclaimer`.

### 4.4 Integration notes
- SDK: **`google-genai`** (the current unified Google Gen AI Python SDK). Auth via `GEMINI_API_KEY` (AI Studio) or Vertex AI credentials on GCP.
- Because generation is now Google-hosted, if the API is deployed on **GCP**, using **Vertex AI** keeps LLM traffic in-network (latency, data-residency, and DPDP/GDPR posture benefits).

---

## 5. API design (backend contract)

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/readings` | Synchronous: image → full reading |
| `POST` | `/v1/jobs` | Async: submit image, returns `job_id` |
| `GET`  | `/v1/jobs/{id}` | Poll async job status/result |
| `POST` | `/v1/features` | CV only — image → feature vector (no LLM) |
| `POST` | `/v1/readings/from-features` | Skip CV — feature vector → reading (great for testing & manual input) |
| `GET`  | `/v1/health` | Health/readiness |

### `POST /v1/readings` request
```json
{
  "image": "<base64 or multipart upload>",
  "hand": "auto | left | right",
  "languages": ["en", "hi"],
  "detail": "short | standard | detailed",
  "model": "gemini-2.5-flash | gemini-2.5-pro"
}
```

### Response
```json
{
  "reading_id": "rd_...",
  "hand": "right",
  "features": { "...feature vector..." },
  "annotated_image_url": "https://.../rd_xxx.png",
  "readings": {
    "en": { "overview": "...", "personality": "...", "relationships": "...", "career": "...", "health": "...", "signs": "..." },
    "hi": { "overview": "...", "personality": "...", "...": "..." }
  },
  "matched_rules": ["heart_long_deep_forked", "..."],
  "confidence": 0.82,
  "disclaimer": "This reading is a traditional interpretation for guidance and entertainment; it is not medical, financial, or predictive advice.",
  "processing_ms": 4210
}
```

### Cross-cutting
- **Auth:** API keys / JWT; per-key **rate limiting** and quota.
- **Privacy:** palm images are biometric-adjacent PII — encrypt at rest, define retention (e.g. auto-delete images after N days), allow "don't store" mode. **This matters legally (GDPR/DPDP Act India).**
- **Observability:** structured logs, per-stage timing, confidence metrics, error taxonomy.
- **Idempotency & versioning:** every reading records `rulebook_version` + `model_version` for reproducibility.

---

## 6. Tech stack

| Concern | Choice | Why |
|---|---|---|
| API framework | **FastAPI** (Python, async) | CV/ML ecosystem is Python; async for LLM I/O |
| Hand landmarks | **MediaPipe Hands** | Solved, free, accurate |
| CV | **OpenCV** + **NumPy** | Line/mount classical processing |
| Line segmentation (prod) | **PyTorch** (U-Net / SAM fine-tune) | Production accuracy |
| Rulebook store | **Postgres** (or SQLite for prototype) | Queryable, versioned |
| LLM | **Gemini 2.5 Flash** (`google-genai` SDK; AI Studio or Vertex AI) | Fast, cheap, strong native Hindi; JSON structured output |
| Image/artifact store | **S3-compatible** | Annotated images, temp uploads |
| Job queue (async) | **Celery + Redis** (or RQ) | Heavy CV off the request path |
| Container | **Docker** | Reproducible deploy |

---

## 7. Data & the real bottleneck

The single biggest risk is **line-detection accuracy**. Research (July 2026) shows this is **less of a bottleneck than first assumed** — a pretrained palmistry model already exists:

- **Start with pretrained weights:** [`yeonsumia/palmistry`](https://github.com/yeonsumia/palmistry) ships a ready 55.5 MB checkpoint (Apache-2.0). No dataset needed to begin.
- **If you must train/fine-tune**, prefer requesting the **CAS_Palm** dataset (5,251 images, manually labeled principal lines) over hand-labeling from scratch.
- **Only if neither suffices**, hand-label:
  - Collect/curate a few hundred palm images (consent + licensing matter).
  - Hand-label lines (tools: CVAT, Label Studio).
  - Fine-tune the U-Net Context Fusion architecture or the yeonsumia weights; iterate.
- Still budget real effort on **validation** — pretrained models trained on lab-quality images often degrade on real user phone photos (lighting, skin tone, angle). Everything downstream depends on the feature vector being right.

**Reference models/datasets:**
| Asset | Type | Detects | Weights/Data | License/Access |
|---|---|---|---|---|
| [yeonsumia/palmistry](https://github.com/yeonsumia/palmistry) | Model + code | Principal palmistry lines | ✅ committed 55.5MB `.pth` | Apache-2.0 |
| [U-Net Context Fusion](https://arxiv.org/abs/2102.12127) | Architecture (paper) | Palm lines (F1 ≈ 99.42%) | ❌ no public weights | Paper only |
| [CAS_Palm](https://pmc.ncbi.nlm.nih.gov/articles/PMC9590507/) | Dataset + model | Labeled principal lines (5,251 imgs) | ⚠️ on request | Research |
| [hgs1217/Palmprint-Segmentation](https://github.com/hgs1217/Palmprint-Segmentation) | Model + code | Palm **region** (ROI only) | ❌ needs ckpt | — |

---

## 8. Build roadmap

**Phase 0 — Reading engine first (fastest proof, no CV risk)**
- Implement `/v1/readings/from-features` + `/v1/features` schema + rulebook (~50 rules) + Claude bilingual generation.
- You can demo authentic EN/HI readings from manually-entered features in days.

**Phase 1 — Prototype CV pipeline**
- MediaPipe landmarks + classical line/mount extraction → feature vector → full `/v1/readings`.
- End-to-end works; accuracy rough.

**Phase 2 — Production accuracy**
- Build labeled dataset, train segmentation model, add attribute analysis (breaks/islands/forks), expand rulebook to several hundred rules with expert review.

**Phase 3 — Hardening**
- Async jobs, auth/rate-limit/quotas, privacy/retention, annotated-image overlays, observability, cost tuning (prompt caching, model tiering).

---

## 9. Suggested repo structure

```
palm_reader/
├── app/
│   ├── api/            # FastAPI routes, schemas, auth, rate-limit
│   ├── pipeline/
│   │   ├── intake.py       # QA / blur / exposure checks
│   │   ├── landmarks.py    # MediaPipe wrapper
│   │   ├── normalize.py    # ROI crop, deskew, CLAHE
│   │   ├── lines.py        # line extraction + attributes
│   │   ├── mounts.py       # mount analysis
│   │   ├── shape.py        # hand-shape / fingers
│   │   └── features.py     # assemble feature vector
│   ├── knowledge/
│   │   ├── rules.json      # Samudrika Shastra rulebook (versioned)
│   │   └── engine.py       # rule matching
│   ├── generation/
│   │   ├── prompts.py      # system prompts (EN/HI, guardrails)
│   │   └── reader.py       # Gemini 2.5 Flash calls, structured output, validation
│   └── models/         # pydantic schemas (FeatureVector, Reading, ...)
├── ml/                 # training scripts, notebooks, dataset tools
├── tests/              # per-stage unit tests + golden feature vectors
├── data/               # (gitignored) images, labels — NEVER commit PII
└── ARCHITECTURE.md     # this file
```

---

## 10. Open decisions to confirm before coding

1. **Cloud vs self-host** for the API and image storage?
2. **Image retention policy** — store for improvement (with consent) or delete immediately?
3. **Rulebook authorship** — do you have a palmistry domain expert to review content, or should the rulebook cite public classical sources?
4. **Hindi generation strategy** — single bilingual call vs English→Hindi localization pass (recommend the latter for quality).
5. **Scale target** — single-user demo, or multi-tenant with quotas from day one?
```
