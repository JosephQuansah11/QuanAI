"""Content synthesis: turn a request into decks, scenes, features and copy.

The offline engine here is deterministic and template-driven but domain-aware,
so a run with no language model still produces coherent, on-topic artifacts.
When a model *is* available the planner asks it first and validates the answer
against these structures, falling back here on any failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .util import keywords

INTENT_KEYWORDS = {
    "deck": {
        "slide", "slides", "deck", "presentation", "powerpoint", "pptx", "pitch",
        "keynote", "briefing", "talk",
    },
    "image": {
        "image", "images", "picture", "art", "artwork", "logo", "poster", "illustration",
        "photo", "graphic", "render", "wallpaper", "thumbnail", "icon",
    },
    "video": {
        "video", "videos", "animation", "reel", "clip", "movie", "film", "short",
        "promo", "trailer", "mp4",
    },
    "desktop": {"desktop", "tkinter", "pyqt", "electron", "windows", "macos", "linux", "gui"},
    "mobile": {"mobile", "android", "ios", "flutter", "react-native", "phone", "appstore"},
    "media_in": {
        "audio", "transcribe", "transcript", "ingest", "podcast", "footage",
        "subtitles", "captions", "recording", "voiceover",
    },
}

DOMAIN_HINTS = {
    "technology": {
        "ai", "software", "cloud", "data", "algorithm", "model", "robot", "cyber",
        "computer", "digital", "automation", "neural", "api", "quantum", "device",
    },
    "business": {
        "business", "market", "startup", "revenue", "sales", "strategy", "customer",
        "growth", "finance", "investment", "product", "brand",
    },
    "education": {
        "education", "school", "student", "learning", "curriculum", "teacher",
        "training", "course", "lesson", "academic",
    },
    "health": {
        "health", "medical", "patient", "clinic", "wellness", "disease", "therapy",
        "nutrition", "mental", "hospital",
    },
    "environment": {
        "climate", "energy", "solar", "wind", "carbon", "sustainability", "water",
        "recycling", "biodiversity", "emissions", "green",
    },
}

TITLE_CASE_EXCEPTIONS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "with", "at", "by",
}

# words that must stay upper-case in titles
ACRONYMS = {
    "ai", "api", "apis", "ui", "ux", "ml", "llm", "llms", "vr", "ar", "iot", "saas",
    "css", "html", "js", "ios", "sql", "gpu", "cpu", "ceo", "cto", "roi", "kpi", "pdf",
    "pptx", "mp4", "usa", "uk", "eu", "it", "hr",
}


def titlecase(text: str) -> str:
    words = re.split(r"\s+", text.strip())
    out = []
    for index, word in enumerate(words):
        if not word:
            continue
        if word.lower() in ACRONYMS and word.islower():
            out.append(word.upper())
        elif index > 0 and word.lower() in TITLE_CASE_EXCEPTIONS:
            out.append(word.lower())
        elif word.isupper() and len(word) <= 4:
            out.append(word)
        else:
            out.append(word[0].upper() + word[1:])
    return " ".join(out)


FRAMEWORK_HINTS = {
    "tkinter": ("tkinter", "python gui", "desktop python"),
    "pyqt6": ("pyqt", "pyqt6", "qtwidgets"),
    "electron": ("electron", "desktop web app"),
    "pwa": ("pwa", "progressive web", "installable web"),
    "flutter": ("flutter", "dart"),
    "react-native": ("react native", "react-native", "reactnative", "expo"),
}

PLATFORM_OF = {
    "tkinter": "desktop",
    "pyqt6": "desktop",
    "electron": "desktop",
    "pwa": "mobile",
    "flutter": "mobile",
    "react-native": "mobile",
}


def detect_framework(request: str) -> tuple[str | None, str | None]:
    """Return (framework, platform) when the request names a stack explicitly."""
    lowered = request.lower()
    for framework, hints in FRAMEWORK_HINTS.items():
        if any(hint in lowered for hint in hints):
            return framework, PLATFORM_OF[framework]
    return None, None


def detect_domain(request: str) -> str:
    raw = set(re.findall(r"[a-z0-9]+", request.lower()))
    best, score = "general", 0
    for domain, hints in DOMAIN_HINTS.items():
        overlap = len(raw & hints)
        if overlap > score:
            best, score = domain, overlap
    return best


MEDIA_NOUNS = {
    "audio", "video", "clip", "footage", "podcast", "recording", "transcript",
    "transcripts", "subtitles", "captions", "voiceover", "mp3", "mp4", "wav", "mov",
}
ANALYSE_VERBS = {
    "analyse", "analyze", "transcribe", "ingest", "summarise", "summarize", "inspect", "examine",
}
CREATE_VERBS = {
    "make", "create", "build", "generate", "produce", "design", "turn", "edit", "record", "film",
}
MEDIA_PATH = re.compile(
    r"[\w./\-]+\.(?:mp3|wav|m4a|flac|ogg|opus|mp4|mov|mkv|webm|avi|png|jpe?g|gif)",
    re.IGNORECASE,
)


def detect_intents(request: str) -> list[str]:
    """Return every artifact family the request asks for."""
    lowered = request.lower()
    raw = set(re.findall(r"[a-z0-9\-]+", lowered))
    media_file = bool(MEDIA_PATH.search(request))

    found: list[tuple[int, str]] = []
    for intent, hints in INTENT_KEYWORDS.items():
        hits = len(raw & hints)
        hits += sum(1 for hint in hints if " " in hint and hint in lowered)
        if hits:
            found.append((hits, intent))
    found.sort(key=lambda pair: (-pair[0], pair[1]))
    intents = [name for _, name in found]

    # explicit stack selection beats a generic "app"
    explicit_framework, explicit_platform = detect_framework(request)
    if explicit_platform and explicit_platform not in intents:
        intents = [i for i in intents if i not in {"desktop", "mobile"}]
        intents.insert(0, explicit_platform)

    if "desktop" in intents and "mobile" in intents:
        mobile_specific = raw & INTENT_KEYWORDS["mobile"]
        intents = [i for i in intents if i not in {"desktop", "mobile"}]
        intents.insert(0, "mobile" if mobile_specific else "desktop")
    elif "app" in raw and not ({"desktop", "mobile"} & set(intents)):
        intents.insert(0, "mobile" if raw & {"android", "ios", "phone", "appstore"} else "desktop")
    if "appstore" in raw and "mobile" not in intents:
        intents.insert(0, "mobile")

    # "analyse this clip" / "transcribe recording.wav" is ingestion work, not production
    analysing = bool(raw & ANALYSE_VERBS)
    media_subject = bool(raw & MEDIA_NOUNS) or media_file
    creating = bool(raw & CREATE_VERBS)
    if (analysing and media_subject) or "media_in" in intents:
        if "media_in" not in intents:
            intents.insert(0, "media_in")
        if analysing and media_subject and not creating:
            intents = [i for i in intents if i not in {"video", "image", "deck"}] or ["media_in"]
    return intents


# format words that carry no subject matter
ARTIFACT_WORDS = (
    "react native app", "react native", "flutter app", "electron app", "mobile app",
    "desktop app", "web app", "tkinter app", "pyqt app",
    "slide deck", "short promo video", "promo video", "short video", "slide", "slides",
    "powerpoint", "pptx", "presentation", "deck", "pitch", "keynote", "briefing",
    "video", "animation", "reel", "clip", "app", "application", "poster", "image",
    "images", "picture", "artwork", "illustration", "logo", "document", "report",
    "short", "program", "tool", "script",
)

FILLER_PREFIXES = (
    "abstract", "minimal", "minimalist", "modern", "simple", "clean", "cool", "nice",
    "beautiful", "professional", "sleek", "bold", "stunning", "quick", "small", "little",
    "complete", "full", "entire", "new",
)


def topic_of(request: str) -> str:
    """Extract the subject of a request, dropping imperative scaffolding."""
    text = request.strip()
    text = re.sub(
        r"^(please\s+)?(can you\s+)?(i\s+(want|need)\s+(you\s+)?to\s+)?"
        r"(build|create|make|generate|produce|design|develop|write|put together|prepare|draft|give me)\s+"
        r"(me\s+)?(an|a|the|some|my)?\s*",
        "",
        text,
        flags=re.I,
    )
    ranked = sorted(ARTIFACT_WORDS, key=len, reverse=True)
    for _ in range(8):
        before = text
        text = re.sub(r"^\d+\s+", "", text)
        for word in ranked:
            stripped = re.sub(r"^" + re.escape(word) + r"\b\s*", "", text, flags=re.I)
            if stripped != text:
                text = stripped
                break
        text = re.sub(
            r"^(about|on|for|covering|explaining|regarding|that|which|of)\b\s*", "", text, flags=re.I
        )
        for filler in FILLER_PREFIXES:
            stripped = re.sub(r"^" + re.escape(filler) + r"\b\s*", "", text, flags=re.I)
            if stripped != text:
                text = stripped
                break
        if text == before:
            break
    text = re.sub(r"\b(please|for me|thanks)\b", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip(" .,:;")
    if len(text) > 110:
        text = text[:110].rsplit(" ", 1)[0]
    return text or "the topic"


def requested_slides(request: str, default: int = 8) -> int:
    """Pick up an explicit slide count from the request, e.g. "a 7 slide deck"."""
    match = re.search(r"(\d{1,2})\s*(?:-|\s)?\s*(?:slide|slides|page|pages|frame|frames)", request, re.I)
    if match:
        return max(4, min(int(match.group(1)), 24))
    match = re.search(r"(\d{1,2})\s*(?:scene|scenes|shot|shots)", request, re.I)
    if match:
        return max(3, min(int(match.group(1)), 10))
    return default


def _fill(template: str, topic: str, kws: list[str]) -> str:
    vocab = kws + ["the topic", "this area", "the work"]
    return (
        template.replace("{topic}", topic)
        .replace("{kw1}", vocab[0])
        .replace("{kw2}", vocab[1] if len(vocab) > 1 else vocab[0])
        .replace("{kw3}", vocab[2] if len(vocab) > 2 else vocab[0])
    )


SECTION_BANK: dict[str, list[tuple[str, list[str]]]] = {
    "technology": [
        ("Why {topic} matters now", [
            "Adoption of {kw1} is moving from experiments into production budgets",
            "Tooling and infrastructure have matured enough to cut integration risk",
            "Teams that start now compound learning while standards are still forming",
        ]),
        ("Core building blocks", [
            "Data layer: capture, clean and govern the inputs {kw1} depends on",
            "Model layer: hosted APIs versus self-hosted {kw2}, chosen on cost and privacy",
            "Product layer: expose the capability through interfaces people already use",
        ]),
        ("How it works end to end", [
            "Ingest: accept text, image, audio and video through one pipeline",
            "Reason: decompose the request into steps that can be checked",
            "Act: call tools that produce real artifacts rather than descriptions",
            "Verify: test every output against an explicit acceptance rule",
        ]),
        ("Benefits to expect", [
            "Faster cycle time from idea to working artifact",
            "Lower marginal cost per task once the pipeline is stable",
            "Fewer handoffs because one system owns the whole workflow",
        ]),
        ("Risks and how to contain them", [
            "Wrong answers: require evidence from real files, not model claims",
            "Cost drift: cap iterations and prefer local execution where possible",
            "Lock-in: keep the provider layer replaceable behind one interface",
        ]),
        ("Implementation roadmap", [
            "Phase 1 - pilot one workflow end to end and record the baseline",
            "Phase 2 - automate verification and add regression checks",
            "Phase 3 - scale to more use cases and publish internal standards",
        ]),
        ("How success is measured", [
            "Task success rate against automated acceptance checks",
            "Median time to a usable artifact",
            "Cost per artifact, and the share produced without manual fixes",
        ]),
    ],
    "business": [
        ("The opportunity", [
            "Demand for {kw1} is growing faster than the supply of reliable solutions",
            "Existing alternatives are expensive, slow or hard to integrate",
            "A focused offer can win on speed and predictability",
        ]),
        ("Who we serve", [
            "Primary segment: teams already spending on {kw1} today",
            "Secondary segment: buyers who avoided the category because of complexity",
            "Explicit anti-persona: customers needing bespoke compliance work",
        ]),
        ("What we offer", [
            "A complete outcome, not a toolkit - the deliverable arrives ready to use",
            "Transparent pricing with a free local-first tier",
            "Onboarding measured in minutes rather than weeks",
        ]),
        ("Economics", [
            "Marginal delivery cost stays low because generation runs locally",
            "Expansion revenue comes from volume and collaboration features",
            "Break-even is reached at modest usage per account",
        ]),
        ("Competitive landscape", [
            "Incumbents bundle {kw1} into large suites with long contracts",
            "New entrants optimise a single step of the workflow",
            "Our wedge is the verified end-to-end artifact",
        ]),
        ("Go-to-market", [
            "Start with one narrow, high-pain workflow and publish real examples",
            "Convert free users through visible output quality",
            "Build partnerships with agencies that need volume",
        ]),
        ("Milestones", [
            "30 days - first ten users with verified outputs",
            "90 days - a repeatable acquisition channel",
            "180 days - sustainable unit economics",
        ]),
    ],
}

SECTION_BANK.update({
    "education": [
        ("Why this belongs in the curriculum", [
            "{topic} connects theory to tools students already use",
            "It rewards experimentation and makes progress visible",
            "The skills transfer to further study and entry-level work",
        ]),
        ("Learning objectives", [
            "Explain the core ideas of {kw1} in plain language",
            "Apply them to one small but complete project",
            "Evaluate results against explicit criteria",
        ]),
        ("Lesson structure", [
            "Warm-up: one provocation question about {topic}",
            "Core: a worked example with think-aloud commentary",
            "Practice: a guided task with checkpoints",
            "Reflection: what changed in your understanding",
        ]),
        ("Assessment", [
            "Formative: short checks after each checkpoint",
            "Summative: a finished artifact judged with a rubric",
            "Self-assessment against the published criteria",
        ]),
        ("Common misconceptions", [
            "That {kw1} needs specialist equipment before you can begin",
            "That correctness is the only measure of quality",
            "That one large attempt beats several small verified ones",
        ]),
        ("Supporting resources", [
            "Starter templates that remove blank-page friction",
            "Worked examples with commentary",
            "A troubleshooting guide for the three most common failures",
        ]),
        ("Next steps", [
            "Pilot with one class or cohort and gather feedback",
            "Refine the rubric using real student work",
            "Publish the module so others can reuse it",
        ]),
    ],
    "health": [
        ("The problem", [
            "Access to reliable information about {kw1} is uneven",
            "Early signals are easy to miss without structured tracking",
            "Follow-through drops when guidance is generic",
        ]),
        ("What the evidence says", [
            "Consistency matters more than intensity for {topic}",
            "Small measurable changes are easier to sustain",
            "Support and feedback improve adherence",
        ]),
        ("A practical approach", [
            "Baseline: record where things stand today",
            "Plan: choose two changes with the largest expected effect",
            "Review: check progress on a fixed schedule",
        ]),
        ("Safety and guardrails", [
            "Never replace professional advice with generic guidance",
            "Escalate red-flag symptoms immediately",
            "Keep personal data minimal and protected",
        ]),
        ("Measuring progress", [
            "One primary metric tracked weekly",
            "Two supporting metrics tracked monthly",
            "A qualitative check-in each cycle",
        ]),
        ("Common pitfalls", [
            "Changing too many variables at once",
            "Judging results before enough data exists",
            "Ignoring sleep, stress and recovery",
        ]),
        ("Next steps", [
            "Define the primary metric today",
            "Schedule the first review",
            "Remove one obstacle that blocks consistency",
        ]),
    ],
})

SECTION_BANK.update({
    "environment": [
        ("The situation", [
            "Pressure on {kw1} is increasing while budgets stay flat",
            "Most waste is invisible because it is never measured",
            "Early action is far cheaper than late remediation",
        ]),
        ("Where the impact comes from", [
            "Energy use in day-to-day operations",
            "Materials and transport embedded in the supply chain",
            "End-of-life handling and disposal",
        ]),
        ("Practical interventions", [
            "Measure first: instrument the biggest flows",
            "Fix the cheapest high-impact losses before adding capacity",
            "Shift demand rather than only adding supply",
        ]),
        ("Co-benefits", [
            "Lower operating cost within the first year",
            "Better resilience to price and supply shocks",
            "Stronger reporting position for stakeholders",
        ]),
        ("Trade-offs", [
            "Upfront capital versus multi-year savings",
            "Land and resource competition between solutions",
            "Measurement gaps that hide real progress",
        ]),
        ("Roadmap", [
            "Baseline audit and target setting",
            "Two quick wins delivered this quarter",
            "Scale what worked and retire what did not",
        ]),
        ("How to verify claims", [
            "Use measured data, not estimates, for headline numbers",
            "Publish the method alongside the results",
            "Track independently of the team delivering the change",
        ]),
    ],
    "general": [
        ("Why {topic} matters", [
            "It affects decisions people make every day",
            "The cost of getting it wrong is higher than it looks",
            "A small amount of structure removes most of the confusion",
        ]),
        ("Key ideas", [
            "Start from the outcome, not the tooling",
            "Make each step checkable before moving on",
            "Prefer the simplest approach that survives contact with reality",
        ]),
        ("How to approach it", [
            "Define the goal in one sentence anyone can repeat",
            "Break it into steps small enough to finish",
            "Check each step against evidence rather than opinion",
        ]),
        ("What good looks like", [
            "Clear, specific and measurable",
            "Reusable by someone who was not in the room",
            "Honest about limits and assumptions",
        ]),
        ("Common mistakes", [
            "Starting with tools instead of the problem",
            "Skipping verification until the end",
            "Adding complexity faster than value",
        ]),
        ("A concrete plan", [
            "Week 1 - scope and baseline",
            "Weeks 2-3 - build the smallest complete version",
            "Week 4 - verify, review and decide what to scale",
        ]),
        ("Next steps", [
            "Write the one-sentence goal",
            "Choose the first artifact to finish",
            "Schedule the verification review",
        ]),
    ],
})


@dataclass
class Slide:
    title: str
    bullets: list[str] = field(default_factory=list)
    notes: str = ""
    kind: str = "content"  # title | content | image | closing
    image: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "bullets": list(self.bullets),
            "notes": self.notes,
            "kind": self.kind,
            "image": self.image,
        }


@dataclass
class DeckPlan:
    title: str
    subtitle: str = ""
    slides: list[Slide] = field(default_factory=list)
    source: str = "offline"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "slides": [s.to_dict() for s in self.slides],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeckPlan":
        slides = []
        for raw in data.get("slides", []):
            slides.append(
                Slide(
                    title=str(raw.get("title", "")),
                    bullets=[str(b) for b in raw.get("bullets", [])],
                    notes=str(raw.get("notes", "")),
                    kind=str(raw.get("kind", "content")),
                    image=str(raw.get("image", "")),
                )
            )
        return cls(
            title=str(data.get("title", "")),
            subtitle=str(data.get("subtitle", "")),
            slides=slides,
            source=str(data.get("source", "offline")),
        )


def deck_plan(request: str, slide_count: int = 8) -> DeckPlan:
    topic = topic_of(request)
    domain = detect_domain(request)
    kws = keywords(request, 6) or [topic]
    bank = SECTION_BANK.get(domain) or SECTION_BANK["general"]
    slide_count = max(4, min(int(slide_count), 24))

    slides: list[Slide] = [
        Slide(
            title=titlecase(topic),
            bullets=[],
            kind="title",
            notes=f"Open with the single sentence: why {topic} is worth 20 minutes of attention.",
        )
    ]
    for index in range(slide_count - 2):
        section_title, bullets = bank[index % len(bank)]
        rendered = [_fill(b, topic, kws) for b in bullets]
        slides.append(
            Slide(
                title=_fill(section_title, topic, kws),
                bullets=rendered,
                notes=(
                    f"Talk track: state the claim ({rendered[0]}), give one concrete example, "
                    "then hand the question back to the room."
                ),
            )
        )
    slides.append(
        Slide(
            title="Next steps",
            bullets=[
                f"Agree the single most valuable next action for {topic}",
                "Assign an owner and a review date",
                "Decide what evidence will prove it worked",
            ],
            notes="Close by asking for one decision, not a round of applause.",
            kind="closing",
        )
    )
    return DeckPlan(
        title=titlecase(topic),
        subtitle=f"A practical briefing - what it is, why it matters and how to move forward",
        slides=slides,
    )


def image_prompts(request: str, count: int = 3) -> list[dict]:
    topic = topic_of(request)
    kws = keywords(request, 8) or [topic]
    palettes = [
        ("deep indigo and amber", "editorial"),
        ("teal and warm sand", "minimal"),
        ("graphite and electric cyan", "technical"),
        ("plum and soft coral", "organic"),
        ("forest green and ivory", "natural"),
        ("midnight blue and rose gold", "luxury"),
    ]
    styles = [
        "geometric abstraction",
        "layered gradient field",
        "isometric composition",
        "flowing ribbon forms",
    ]
    prompts = []
    for index in range(max(1, int(count))):
        palette, mood = palettes[index % len(palettes)]
        style = styles[index % len(styles)]
        prompts.append(
            {
                "prompt": (
                    f"{style} representing {topic}, {palette} palette, {mood} mood, "
                    "clean negative space, high contrast, no text"
                ),
                "style": style,
                "palette": palette,
                "seed": index,
                "subject": topic,
                "keywords": kws[:5],
            }
        )
    return prompts


def video_plan(request: str, scenes: int = 6, seconds_per_scene: float = 2.5) -> dict:
    topic = topic_of(request)
    kws = keywords(request, 8) or [topic]
    beats = [
        ("Opening question", f"What if {topic} took minutes instead of weeks?"),
        ("The problem", "Most of the time is lost before the real work starts"),
        ("The approach", "One loop: plan, act, verify, repeat"),
        ("What it produces", "Real files you can open, not descriptions of files"),
        ("The evidence", "Every claim is checked against a measurable rule"),
        ("The invitation", "Start with one workflow and measure it"),
    ]
    chosen = beats[: max(3, min(int(scenes), len(beats)))]
    prompts = image_prompts(request, len(chosen))
    timeline = []
    for index, (beat, caption) in enumerate(chosen):
        timeline.append(
            {
                "beat": beat,
                "caption": caption,
                "image_prompt": prompts[index]["prompt"],
                "palette": prompts[index]["palette"],
                "seconds": float(seconds_per_scene),
                "seed": index,
            }
        )
    return {
        "title": titlecase(topic),
        "subtitle": "A short visual explainer",
        "topic": topic,
        "keywords": kws[:6],
        "scenes": timeline,
        "duration": round(len(timeline) * float(seconds_per_scene), 1),
    }


FRAMEWORKS = {
    "desktop": {
        "tkinter": {
            "label": "Python + Tkinter",
            "requires": [],
            "run": "python3 main.py",
            "why": "ships with Python, zero install, runs on Windows/macOS/Linux",
        },
        "pyqt6": {
            "label": "Python + PyQt6",
            "requires": ["PyQt6"],
            "run": "python3 main.py",
            "why": "native-looking widgets when PyQt6 can be installed",
        },
        "electron": {
            "label": "Electron (HTML/CSS/JS)",
            "requires": ["node"],
            "run": "npm install && npm start",
            "why": "web UI with full desktop shell access",
        },
    },
    "mobile": {
        "pwa": {
            "label": "Installable PWA",
            "requires": [],
            "run": "python3 -m http.server 8000 (then open on the phone)",
            "why": "installs to a phone home screen, works offline, no SDK needed",
        },
        "flutter": {
            "label": "Flutter (Dart)",
            "requires": ["flutter"],
            "run": "flutter run",
            "why": "one codebase for Android and iOS",
        },
        "react-native": {
            "label": "React Native (Expo)",
            "requires": ["node"],
            "run": "npm install && npx expo start",
            "why": "JavaScript-native apps with a large ecosystem",
        },
    },
}


def pascal_case(text: str, max_words: int = 3) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", text) if p][:max_words]
    if not parts:
        return "QuanApp"
    name = "".join(p[0].upper() + p[1:] for p in parts)
    if not name[0].isalpha():
        name = "App" + name
    return name[:32]


def app_plan(request: str, platform: str = "desktop", framework: str | None = None) -> dict:
    platform = platform if platform in FRAMEWORKS else "desktop"
    options = FRAMEWORKS[platform]
    framework = framework if framework in options else ("tkinter" if platform == "desktop" else "pwa")
    topic = topic_of(request)
    kws = keywords(request, 6)
    name = pascal_case(topic)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    noun = kws[0] if kws else "item"

    features = [
        {"id": "add", "title": "Add", "detail": f"Capture a new {noun} in under three seconds"},
        {"id": "list", "title": "List", "detail": "See everything collected, newest first, with an empty state"},
        {"id": "complete", "title": "Complete", "detail": "Mark an entry done and keep the history"},
        {"id": "search", "title": "Search", "detail": "Filter entries by text as you type"},
        {"id": "stats", "title": "Progress", "detail": "A live count of open versus finished entries"},
        {"id": "persist", "title": "Persistence", "detail": "Data survives a restart, stored locally with no account"},
    ]
    description = (
        f"{name} is a {platform} application for {topic}. It focuses on the core loop - "
        "capture, review, complete - and keeps all data on the device."
    )
    return {
        "name": name,
        "slug": slug,
        "platform": platform,
        "framework": framework,
        "framework_label": options[framework]["label"],
        "run": options[framework]["run"],
        "description": description,
        "topic": topic,
        "features": features,
        "screens": ["Main", "Add entry", "Settings/About"],
        "keywords": kws,
    }


def analysis_outline(subject: str, media_kind: str) -> dict:
    return {
        "title": f"Analysis: {subject}",
        "sections": [
            ("What was supplied", f"Technical facts about the {media_kind} file."),
            ("Structure", "How the material is organised over time or space."),
            ("Observations", "What the extracted signals show."),
            ("Limits", "What these local methods cannot tell you."),
            ("Next steps", "How to go further with a capable model."),
        ],
    }
