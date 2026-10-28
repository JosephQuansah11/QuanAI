# QuanAI

A self-checking, multimodal agent that turns one sentence into finished artifacts:
**PowerPoint decks, images, videos, desktop apps, mobile apps, and analysis of audio /
video / image input** — then proves the work is real before calling it done.

It is built to run with **no paid API and no GPU**. Planning, content, rendering,
encoding and verification all run locally. A language model is optional: plug in
Ollama / LM Studio / vLLM / any OpenAI-compatible endpoint and the agent uses it for
richer drafting and critique; leave it out and the deterministic engine takes over.

```bash
quanai run "create a 7 slide deck about solar energy adoption in cities"
quanai run "build a desktop app for tracking my daily habits"
quanai run "make a short promo video about our AI agent"
quanai run "analyse this clip and report on its structure" --input samples/clip.mp4
```

Every run leaves you with the artifacts **plus** `REPORT.md` and `manifest.json`
containing the evidence for each requirement.

---

## Why it is different

Most agent demos stop when the model says it succeeded. QuanAI does not accept that:

| Principle | How it is implemented |
| --- | --- |
| Requirements are explicit | Every request is decomposed into `Requirement` objects, each carrying at least one machine-checkable rule. |
| Checks are executed, not assumed | `pptx_valid`, `image_valid`, `video_valid`, `scaffold_files`, `python_compiles`, `js_syntax`, `markdown_report` … actually open and measure the file. |
| Tasks can't fake success | A task is only `done` when its own checks pass; a tool returning `ok` is not enough. |
| Failure produces a different attempt | The critic queues a repair task with a **changed strategy** (alternate theme, simpler encode, fallback framework), never a blind retry of the same call. |
| Nothing is lost | State is written to disk after every step, so an interrupted run resumes exactly where it stopped. |
| The model cannot overrule evidence | An LLM may add insight and draft content, but a failed check stays failed. |
| It learns between runs | Failures are stored as `lessons` (failure signature → advice) and validated plan shapes are reused for similar requests. |

---

## Install

```bash
pip install -e .            # core: python-pptx, pillow, numpy, imageio-ffmpeg
pip install -e ".[dev]"     # + pytest
```

Optional extras:

```bash
pip install -e ".[asr]"     # faster-whisper: real speech-to-text for audio/video
pip install -e ".[vision]"  # opencv-headless (not required; numpy/Pillow path is default)
```

Check the machine before the first run:

```bash
quanai doctor
```

```text
Capabilities
  [ok] python           /usr/bin/python3
  [ok] node             /usr/local/bin/node
  [ok] ffmpeg           .../imageio_ffmpeg/binaries/ffmpeg-linux-x86_64
  [ok] pillow           True
  [ok] numpy            True
  [ok] python-pptx      True
  [--] faster-whisper   not available

Language model
  offline deterministic engine (no model configured)
```

---

## Quick start

```bash
quanai run "create a 7 slide deck about solar energy adoption in cities"
```

```text
Objective obj-3e6c3a41  [done]  100.0% verified
Request: Create a 7 slide powerpoint deck about solar energy adoption in cities
Provider: offline (deterministic)   Iterations: 1   Time: 1.5s

Requirements:
  [ok  ] R.deck: A PowerPoint deck about solar energy adoption in cities exists ...
          + deck has 8 slides and 1346 characters of text
  [ok  ] R.deck.notes: Most slides carry speaker notes so the deck can be presented
          + 8 of 8 slide(s) have speaker notes (need 4)
  [ok  ] R.deck.art: Cover artwork was generated locally for the deck
          + 1600x900 png image, detail std 33.8

Artifacts:
  plans/deck.json  (3.6KB)
  assets/deck-art-1.png  (3.0MB)
  assets/deck-art-2.png  (3.0MB)
  deck.pptx  (50.3KB)

Report: workspace/obj-3e6c3a41/REPORT.md
```

Inspect the plan without running anything:

```bash
quanai plan "build a mobile app for habit tracking"
```

---

## What it can build

### 1. PowerPoint decks (`.pptx`)

Real 16:9 decks with a title slide, bulleted content slides, speaker notes on every
slide, page numbers, a chosen theme and optional full-bleed artwork.

```bash
quanai run "a 10 slide deck about quantum computing for investors"
```

- Themes: `midnight`, `paper`, `forest`, `plum`
- Pipeline: `plan.deck` (writes `plans/deck.json`) → `image.variations` (cover art) → `slides.create`
- Verified by: slide count, text volume, subject term present, notes per slide

### 2. Images

Deterministic procedural art: seeded plasma field, palette wash, layered geometry,
grain and vignette. Same prompt + seed always produces the same image, so outputs are
reproducible and reviewable.

```bash
quanai run "generate an abstract poster image about deep ocean exploration"
```

- Backends: `procedural` (default, offline), `sd` (local Stable Diffusion WebUI), `pollinations` (free, needs network)
- Select with `--image-backend` or `QUANAI_IMAGE_BACKEND`
- Verified by: decodable image, minimum size, non-blank detail variance

### 3. Video (`.mp4`)

Storyboard → caption cards → H.264 slideshow with Ken Burns motion, optional audio.
Uses the ffmpeg binary bundled with `imageio-ffmpeg`, so no system install is needed.

```bash
quanai run "make a short promo video about our AI agent"
```

- Pipeline: `plan.video` → `video.storyboard` (art + captions per scene) → `video.from_images`
- Verified by: video stream present, duration, width
- Falls back to an animated GIF if ffmpeg is truly unavailable

### 4. Desktop applications

```bash
quanai run "build a desktop app for tracking my daily habits"
```

| Framework | Run | Notes |
| --- | --- | --- |
| `tkinter` (default) | `python3 main.py` | zero dependencies, Windows/macOS/Linux |
| `pyqt6` | `python3 main.py` | native widgets, needs `pip install PyQt6` |
| `electron` | `npm install && npm start` | HTML/CSS/JS with a secure preload bridge |

Each generated app is complete: add / list / complete / search / progress / local
persistence, empty states, and a README. Verified by file presence **and** a real
`python_compile` or `node --check`.

### 5. Mobile applications

```bash
quanai run "create a mobile app for habit tracking"          # installable PWA
quanai run "build a react native app for daily habits"      # Expo
quanai run "create a flutter app for tracking workouts"     # Flutter
```

| Framework | Run | Notes |
| --- | --- | --- |
| `pwa` (default) | `python3 -m http.server 8000` | installs to the home screen, offline service worker, real icons |
| `flutter` | `flutter run` | Dart widget tree + `pubspec.yaml` + widget test |
| `react-native` | `npm install && npx expo start` | Expo app with AsyncStorage persistence |

The PWA is the default because it can be verified and used here without an SDK.

### 6. Audio / video / image understanding

```bash
quanai ingest samples/clip.mp4          # one-shot analysis
quanai run "analyse this clip and report on its structure" --input samples/clip.mp4
```

Local, free signal analysis written to `observations.json` plus a written report:

- **Image**: dimensions, brightness, contrast, edge density, dominant colours, perceptual hash, optional model caption, description text
- **Video**: duration/codec/fps, sampled frames, contact sheet, scene-change scores, audio track analysis
- **Audio**: duration, peak/RMS, silence ratio, speech-like segments, waveform and spectrogram PNGs
- **Speech**: transcript when a local engine is available (`faster-whisper`, `whisper` CLI, or your own command via `QUANAI_ASR_CMD`)

---

## How the loop works

```text
request
  │
  ├─ think      capability probe → candidate toolchains → scored choice (stored in the trace)
  ├─ plan       requirements (with checks) + dependency-ordered tasks
  │
  ▼
iteration ──▶ execute every runnable task (tool → real file)
  │             └─ each task runs its own checks before being marked done
  ├─ verify    re-run every requirement check (the constant check)
  ├─ critique  pass/fail per requirement → repair tasks with a new strategy
  │             └─ lessons written to memory for next time
  └─ repeat    until everything required is verified, or escalate/block honestly
  │
  ▼
REPORT.md + manifest.json + artifacts
```

The loop stops for exactly three reasons:

1. **done** — every required check passed against the real files;
2. **blocked** — a capability or input is missing (it tells you what to supply);
3. **budget** — iterations ran out with requirements still unverified (the report lists them).

### Requirement checks

| Check | What it actually does |
| --- | --- |
| `file_exists` | resolves the path inside the workspace (glob supported), refuses escapes |
| `file_min_bytes` | opens the file and compares size |
| `text_contains` | reads the file, requires phrases and a character floor |
| `no_placeholders` | scans for `TODO`, `FIXME`, `lorem ipsum`, `NotImplementedError` |
| `json_valid` | parses JSON and asserts required keys |
| `zip_valid` | opens the archive, tests members, checks required entries |
| `pptx_valid` | opens with python-pptx, counts slides, characters and required terms |
| `pptx_notes` | counts slides that carry real speaker notes |
| `image_valid` | decodes with Pillow, checks size, rejects flat/blank images by variance |
| `video_valid` | decodes with ffmpeg, checks video stream, duration, width, audio |
| `audio_valid` | decodes with ffmpeg, checks audio stream and duration |
| `python_compiles` | compiles every `.py` in the target with the real parser |
| `js_syntax` | runs `node --check` on each `.js` file |
| `scaffold_files` | required entrypoints exist, are non-trivial, alternative groups satisfied |
| `media_report` | observations JSON parsed and required groups present |
| `markdown_report` | heading count and content volume |

Unknown checks are rejected at plan time, and a check that raises is recorded as a
failure instead of crashing the run.

---

## Language models (optional)

QuanAI works with no model at all. When one is available it is used for drafting deck
content, storyboards, app feature lists, and for deeper critique.

| Provider | How to enable |
| --- | --- |
| Ollama | `ollama serve` → `export QUANAI_LLM_PROVIDER=ollama QUANAI_MODEL=llama3.1` |
| LM Studio | start the local server → `QUANAI_LLM_PROVIDER=lmstudio QUANAI_MODEL=local-model` |
| vLLM / llama.cpp | `QUANAI_LLM_PROVIDER=vllm QUANAI_BASE_URL=http://127.0.0.1:8000/v1` |
| OpenAI or compatible | `QUANAI_API_KEY=... QUANAI_MODEL=gpt-4o-mini` |
| Fully offline | `QUANAI_LLM_PROVIDER=offline` (also the default when nothing is found) |

`auto` probes `127.0.0.1:11434`, `:1234`, `:8000`, `:8080` with a sub-second timeout,
then falls back to a hosted key, then to the deterministic engine. Model output is
validated: if it is not usable JSON, or the slides are too thin, the deterministic
engine fills the gap and the deck still verifies.

Every model call is optional. A raising provider never breaks a run — a test asserts it.

---

## Command line

```text
quanai run "<request>" [--input FILE ...] [--id ID] [--max-iterations N] [--json]
quanai plan "<request>"          # show requirements + task graph, run nothing
quanai status [--limit N]        # list objectives across runs
quanai show <id>                 # dump one objective as JSON
quanai resume <id>               # continue an interrupted objective
quanai ingest <file>             # analyse one media file
quanai tools                     # list the tool surface
quanai doctor [--all]            # capabilities + model probe
```

Global flags: `--workspace DIR`, `--home DIR`, `--provider NAME`, `--image-backend NAME`.

Exit codes: `0` verified, `2` finished unverified, `1` error.

---

## Python API

```python
from quanai import Agent, Config

config = Config.from_env()
agent = Agent(config)                     # provider auto-detected, offline if none
report = agent.run("create a 6 slide deck about urban cycling")

print(report.status, report.completion, report.verified)
for artifact in report.artifacts:
    print(artifact["path"], artifact["bytes"])
print(open(report.report_path).read())
```

Continue a run later, even in a new process:

```python
Agent(Config.from_env()).resume(report.objective_id)
```

Use your own tools or a custom model:

```python
from quanai.core.agent import Agent
from quanai.llm.base import CallableProvider
from quanai.tools import build_default_registry

registry = build_default_registry()

@registry.tool("company.publish", "publish an artifact", schema={
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
})
def publish(ctx, path):
    ...

provider = CallableProvider(lambda messages, json_mode=False: '{"ok": true}')
Agent(Config.from_env(), provider=provider, registry=registry).run("...")
```

---

## Layout

```text
quanai/
  config.py          environment-driven settings, path normalisation
  content.py         intent detection, topic extraction, deck/video/app content engines
  probe.py           media probing (Pillow + ffmpeg -i parsing)
  core/
    task.py          Objective / Requirement / Task / Check / Evidence + scheduling
    verify.py        check registry and Verifier (evidence collection)
    planner.py       capability-aware planning + plan reuse + reasoning trace
    critic.py        evidence-driven critique, repair strategies, lessons
    agent.py         the loop, task execution, reporting
    ledger.py        crash-safe objective storage + event log
    memory.py        searchable memory, lessons, plan templates
  llm/
    base.py          provider protocol, multimodal messages, JSON extraction
    openai_compat.py stdlib chat client (OpenAI, Ollama, vLLM, LM Studio, ...)
    router.py        auto-detection with local-first ordering
  tools/
    registry.py      tool specs, schemas, sandboxed execution, artifacts
    fs.py code.py    filesystem, python/node/shell, zip
    slides.py        16:9 themed .pptx builder
    images.py        procedural art + SD + Pollinations backends
    video.py         storyboard cards, ffmpeg encoding, audio stills
    media.py         multimodal ingestion and local signal analysis
    apps.py          tkinter / PyQt6 / Electron / PWA / Flutter / React Native scaffolds
    plan.py report.py  structured planning artifacts and markdown writing
  cli.py             command line interface
```

---

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `QUANAI_WORKSPACE` | `./workspace` | where objective folders and artifacts are written |
| `QUANAI_HOME` | `./.quanai` | ledger, event logs, memory, lessons |
| `QUANAI_LLM_PROVIDER` | `auto` | `auto`, `offline`, `ollama`, `lmstudio`, `vllm`, `openai`, custom |
| `QUANAI_BASE_URL` / `QUANAI_API_KEY` / `QUANAI_MODEL` | – | endpoint, key and model |
| `QUANAI_IMAGE_BACKEND` | `auto` | `procedural`, `sd`, `pollinations` |
| `QUANAI_ASR_CMD` | – | custom transcription command, e.g. `whisper {audio} --output_dir {output}` |
| `QUANAI_ALLOW_SHELL` | `0` | enables `code.shell` (still denies destructive patterns) |
| `QUANAI_ALLOW_NETWORK` | `0` | allows remote image generation |
| `QUANAI_MAX_ITERATIONS` | `6` | loop budget per objective |
| `QUANAI_MAX_ATTEMPTS` | `3` | retries per task |

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest             # 95 tests, all offline (about 15s)
```

The suite covers the domain model, every check, every tool's real output, planning
invariants (no unknown tools or checks, resolvable references), provider fallbacks,
repair behaviour, resume, and the CLI.

---

## Honest limits

- The offline content engine is template-driven and domain-aware, not a writer. With a
  language model installed the prose is noticeably better; the structure is the same.
- Procedural images are abstract art generated from a prompt, not photorealistic
  text-to-image. Point `QUANAI_IMAGE_BACKEND=sd` at a local Stable Diffusion WebUI for that.
- Video is a captioned slideshow with motion, not generative footage.
- Speech-to-text needs a local engine; without one you still get waveform, spectrogram,
  VAD segments and structure — and the report says the transcript is unavailable.
- Flutter and React Native projects are generated and structurally verified, but running
  them needs their SDKs; the PWA path is fully usable and installable today.
- Generated apps are working starters (one screen, persisted data), not production
  products. They are meant to be opened, run and extended.
