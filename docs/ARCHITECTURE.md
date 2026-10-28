# Architecture

QuanAI is a loop around four durable concepts: **objectives**, **requirements**,
**tasks** and **evidence**. Everything else is replaceable.

```
                     ┌──────────────────────────────────────────┐
   user request ───▶ │ Planner: intents → requirements → tasks  │
                     │ (capability probe + reasoning trace)     │
                     └───────────────┬──────────────────────────┘
                                     ▼
                     ┌──────────────────────────────────────────┐
                     │ Objective (persisted after every step)   │
                     │  requirements[]  tasks[]  lessons[]      │
                     └───────────────┬──────────────────────────┘
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        │ Agent loop (bounded by max_iterations)                  │
        │  1. drain ready tasks → ToolRegistry → real files       │
        │  2. Verifier re-runs every requirement check            │
        │  3. Critic compares evidence → repair tasks / lessons   │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
                    REPORT.md · manifest.json · artifacts
```

## 1. Domain model (`core/task.py`)

| Type | Purpose |
| --- | --- |
| `Check` | one machine-verifiable rule: `kind` + `params` |
| `Requirement` | what the user asked for, plus the checks that prove it, plus `Evidence` |
| `Task` | one tool call: `tool`, `args`, `depends_on`, `satisfies`, own `checks`, attempt counter |
| `Objective` | the whole job: request, requirements, tasks, iteration, thoughts, lessons |

Scheduling is a fixpoint over `depends_on`: a task is `ready` when all dependencies are
`done`, and `blocked` when one failed. `verified()` requires every required requirement
to pass **and** no *blocking* failure — a failed attempt that a later repair superseded
does not block completion.

## 2. Verification (`core/verify.py`)

Checks are registered functions with the signature
`(ctx: VerifyContext, **params) -> (ok, detail, observed)`.
`Verifier.verify_objective` runs every requirement check, appends `Evidence`, and returns
a pass/fail map. Requirements only ever look at their **latest** evidence, so a repair
that overwrites the artifact correctly flips the requirement back to passing.

Design rules:

- checks must open the real file (Pillow, python-pptx, ffmpeg, zipfile, `compile()`);
- a raising check is a failure, never a crash;
- unknown check kinds are rejected at plan time;
- paths are resolved inside the objective workspace and escapes are refused.

## 3. Planning (`core/planner.py`)

`think()` probes the machine and records the candidate toolchains, the chosen path and
the rationale — this is the reasoning trace that ends up in `REPORT.md`.

Track builders emit requirements and tasks for: deck, image, video, desktop app, mobile
app, media ingestion and plain documents. A request can produce several tracks at once
("a deck and an app"). Task arguments may reference earlier results with
`${task.id::dotted.path}`; unresolved references fail the task with a precise message.

A validated plan shape from memory is reused for a similar request
(`source: memory`), with the stored request text rewritten into every argument so the
new objective cannot accidentally rebuild the old one.

## 4. Execution (`core/agent.py`)

Per iteration:

1. drain **all** runnable tasks (so a chain finishes inside one iteration);
2. after each task, run the task's own checks — `ok` from a tool is not acceptance;
3. re-verify the objective (the constant check);
4. ask the critic what to do next.

Failure handling: a task is retried up to `max_attempts`, then marked failed. The critic
then creates a **new** task id (`deck.build-r2`) with a changed strategy and the same
output path — the path is the contract with the requirement checks, so it must be
overwritten rather than renamed.

The loop persists the objective before and after every step, which is what makes
`quanai resume <id>` work in a fresh process.

## 5. Critic (`core/critic.py`)

Inputs: verification results, failed tasks, iteration number.
Outputs: `Decision(complete, gaps, revisions, lessons, escalate)`.

Repair strategies by tool:

| Tool | Changed on repair |
| --- | --- |
| `slides.create` | next light palette (paper → sky → mint → sand), slides resolved from the plan file |
| `video.from_images` | zoom off, smaller frame size, longer per-image duration |
| `video.storyboard` | smaller cards |
| `image.*` | procedural backend, different style, more variants |
| `app.scaffold` | PWA fallback for SDK-dependent frameworks, otherwise tkinter |
| `code.zip`, `report.write` | repackage / rewrite with explicit sections |

Two requirements repaired by the same artifact collapse into one revision with both
`requirement` ids in `satisfies`.

Escalation: when nothing new can be tried, or the same failure repeats, the objective
becomes `blocked` with a human-readable reason (for example: a missing input file).

## 6. Memory (`core/memory.py`)

- **items**: append-only observations, keyword/IDF scored, used for recall;
- **lessons**: `(tool, failure signature) → advice`, consulted before a retry;
- **plans**: validated plan shapes for reuse.

This is the "learning" surface: it is small, inspectable and deterministic — no opaque
weights, no training run, and it never overrides a check.

## 7. Tools (`tools/`)

Every tool declares a JSON-schema-ish contract, a category, its optional dependencies
and a handler `(ctx, **args) -> ToolResult`. `ToolRegistry.execute` validates required
arguments, sandboxes paths to the workspace, times the call and converts exceptions into
structured failures with a traceback tail.

Artifacts are first-class: `Artifact(path, kind, mime, bytes, meta)` is what reports,
manifests and the critic talk about.

## 8. Determinism

Given the same request, the same machine and no model, QuanAI produces byte-stable
images (seeded renderer), stable plans and stable task ids. That makes regressions
visible and tests meaningful.

## 9. Look and feel: light by default

Every artifact QuanAI produces is light unless a user explicitly asks otherwise:

| Surface | Where the palette lives | Default |
| --- | --- | --- |
| Decks | `tools/slides.py` → `THEMES`, `LIGHT_THEMES`, `DEFAULT_THEME` | `paper` (also `sky`, `mint`, `sand`) |
| Desktop / mobile UI | `tools/apps.py` → `TKINTER_MAIN`, `PYQT_MAIN`, `APP_CSS`, Flutter/RN styles | near-white `#f7f8fa` surfaces, `#111827` ink, `#2563eb` accent |
| Video cards | `tools/video.py` → `compose_card` | light veil + dark caption text |
| Generated art | `tools/images.py` → `LIGHT_STOPS` | pastel palette, gentle vignette |
| Analysis visuals | `tools/media.py` | paper background waveform, inverted spectrogram |
| App icons / manifest | `tools/apps.py` → `_icon`, manifest builder | light icon, `#f7f8fa` splash |

Dark palettes (`midnight`, `forest`, `plum`) and dark art (`light: false`) still exist but
are opt-in. Choosing a palette is explicit: `quanai run "..." --theme sky`, or
`QUANAI_THEME=sand`. An unknown theme name falls back to `paper` rather than failing.

The critic rotates through the light palettes when repairing a deck
(`_next_light_theme`), so a repair always changes the look without going dark.

`tests/test_light_theme.py` enforces all of this: it renders a real deck, all six app
frameworks, storyboard cards, icons and analysis images, then asserts luminance and
WCAG contrast (body ≥ 7:1, muted and accent ≥ 4.5:1).
