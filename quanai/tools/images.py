"""Image generation that works with no API key and no GPU.

The default backend is a local procedural renderer: a seeded plasma field,
gradient wash and layered geometry, so the same prompt always yields the same
image.  Optionally it will delegate to a local Stable Diffusion WebUI or the
free Pollinations endpoint when those are available.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import math
import urllib.parse
import urllib.request
from pathlib import Path

from ..errors import MissingDependency, ToolError
from ..util import digest, keywords
from .registry import Artifact, ToolContext, ToolResult, artifact

MIME = "image/png"

STYLE_TUNING = {
    "editorial": {"waves": 3, "shapes": 5, "grain": 0.05, "contrast": 1.12},
    "minimal": {"waves": 2, "shapes": 3, "grain": 0.03, "contrast": 1.05},
    "technical": {"waves": 4, "shapes": 8, "grain": 0.06, "contrast": 1.2},
    "organic": {"waves": 5, "shapes": 6, "grain": 0.05, "contrast": 1.08},
    "luxury": {"waves": 3, "shapes": 4, "grain": 0.04, "contrast": 1.15},
}


def _rng(seed_parts, seed: int = 0):
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency("image.generate", "numpy", "pip install numpy") from exc
    fingerprint = int(hashlib.sha256(str(seed_parts).encode()).hexdigest()[:8], 16)
    return np.random.default_rng((fingerprint + int(seed) * 7919) % (2**32))


# Light palettes are the default: airy tints for the wash, one deeper hue for
# structure. Dark stops are only used when somebody explicitly asks for them.
LIGHT_STOPS = (
    (0.0, 0.10, 0.99),    # near-white tint
    (1.0, 0.26, 0.95),    # pale wash
    (2.1, 0.44, 0.84),    # mid tint for shape fills
    (3.2, 0.70, 0.46),    # deeper hue for lines and structure
)
DARK_STOPS = (
    (0.0, 0.85, 0.30),
    (1.0, 0.75, 0.62),
    (2.1, 0.62, 0.88),
    (3.2, 0.45, 0.97),
)


def _palette(rng, style: str, light: bool = True) -> list[tuple[int, int, int]]:
    base = float(rng.random())
    spread = {
        "editorial": 0.22, "minimal": 0.12, "technical": 0.30,
        "organic": 0.18, "luxury": 0.16,
    }.get(style, 0.2)
    stops = LIGHT_STOPS if light else DARK_STOPS
    colors = []
    for offset, saturation, value in stops:
        hue = (base + offset * spread) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def render_procedural(
    prompt: str,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    style: str = "editorial",
    light: bool = True,
):
    """Render a deterministic abstract composition for the prompt."""
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageDraw, ImageFilter  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            "image.generate", "pillow+numpy", "pip install pillow numpy"
        ) from exc

    width, height = max(64, int(width)), max(64, int(height))
    tuning = STYLE_TUNING.get(style, STYLE_TUNING["editorial"])
    rng = _rng(f"{prompt}|{style}", seed)
    palette = _palette(rng, style, light)

    ys, xs = np.mgrid[0:height, 0:width].astype("float32")
    u = xs / max(width - 1, 1)
    v = ys / max(height - 1, 1)

    # gradient wash between two palette stops on a diagonal
    angle = float(rng.uniform(0.15, 0.85))
    t = np.clip(u * angle + v * (1 - angle), 0.0, 1.0)[..., None]
    c0 = np.array(palette[0], dtype="float32")
    c1 = np.array(palette[1], dtype="float32")
    field = c0 * (1 - t) + c1 * t

    # plasma: sum of sine waves with random phases
    plasma = np.zeros((height, width), dtype="float32")
    for _ in range(tuning["waves"]):
        fx = rng.uniform(0.8, 3.6)
        fy = rng.uniform(0.8, 3.6)
        phase = rng.uniform(0, 2 * math.pi)
        plasma += np.sin((u * fx + v * fy) * math.pi * 2 + phase)
    plasma = plasma / max(tuning["waves"], 1)
    plasma = (plasma + 1.0) / 2.0

    accent = np.array(palette[2], dtype="float32")
    highlight = np.array(palette[3], dtype="float32")
    mask = np.clip((plasma - 0.45) * 2.2, 0.0, 1.0)[..., None]
    field = field * (1 - mask * 0.55) + accent * (mask * 0.55)
    glow = np.exp(-(((u - 0.62) ** 2 + (v - 0.38) ** 2) / 0.07))[..., None]
    field = field + highlight * glow * 0.35

    image = Image.fromarray(np.clip(field, 0, 255).astype("uint8"), mode="RGB")

    # layered geometry: soft rings, arcs and angular shards
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index in range(tuning["shapes"]):
        alpha = int(rng.integers(40, 112)) if light else int(rng.integers(26, 78))
        color = palette[(index + 1) % len(palette)] + (alpha,)
        cx = float(rng.uniform(-0.1, 1.1)) * width
        cy = float(rng.uniform(-0.1, 1.1)) * height
        span = float(rng.uniform(0.12, 0.55)) * min(width, height)
        kind = int(rng.integers(0, 3))
        if kind == 0:
            draw.ellipse([cx - span, cy - span, cx + span, cy + span], outline=color, width=max(2, int(span * 0.06)))
        elif kind == 1:
            draw.arc(
                [cx - span, cy - span, cx + span, cy + span],
                start=float(rng.uniform(0, 360)), end=float(rng.uniform(0, 360)),
                fill=color, width=max(2, int(span * 0.05)),
            )
        else:
            points = [
                (cx + float(rng.uniform(-span, span)), cy + float(rng.uniform(-span, span)))
                for _ in range(3)
            ]
            draw.polygon(points, fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.6, 2.4))))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    # film grain + vignette so the result reads as art, not as a gradient.
    # Light art needs a much gentler vignette or the paper turns grey at the edges.
    arr = np.asarray(image.convert("RGB"), dtype="float32")
    grain = rng.normal(0.0, 255.0 * tuning["grain"], (height, width, 1))
    arr = arr + grain
    falloff = 0.16 if light else 0.55
    floor = 0.72 if light else 0.25
    vignette = 1.0 - falloff * (((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2.2) ** 1.4
    arr = arr * np.clip(vignette, floor, 1.0)[..., None]
    contrast = tuning["contrast"] if not light else 1.0 + (tuning["contrast"] - 1.0) * 0.45
    arr = (arr - 127.5) * contrast + 127.5
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")


def render_pollinations(prompt: str, width: int, height: int, seed: int, timeout: float = 60.0):
    """Free, key-less remote generator (needs network access)."""
    from PIL import Image  # type: ignore

    query = urllib.parse.quote(prompt[:900])
    url = (
        f"https://image.pollinations.ai/prompt/{query}"
        f"?width={int(width)}&height={int(height)}&seed={int(seed)}&nologo=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "QuanAI/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload.startswith(b"\x89PNG") and not payload.startswith(b"\xff\xd8"):
        raise ToolError("pollinations returned a non-image payload")
    return Image.open(io.BytesIO(payload)).convert("RGB")


def render_sd(prompt: str, width: int, height: int, seed: int, base_url: str = "http://127.0.0.1:7860", timeout: float = 180.0):
    """Local Stable Diffusion WebUI (AUTOMATIC1111) txt2img."""
    import base64
    import json

    from PIL import Image  # type: ignore

    body = json.dumps(
        {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "steps": 22,
            "sampler_name": "DPM++ 2M",
            "cfg_scale": 6.5,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/sdapi/v1/txt2img",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    images = data.get("images") or []
    if not images:
        raise ToolError("stable diffusion returned no images")
    raw = images[0].split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

def generate(
    ctx: ToolContext,
    path: str,
    prompt: str,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    style: str = "editorial",
    backend: str = "auto",
    light: bool = True,
):
    """Generate one image, trying backends in the configured order."""
    target = ctx.path(path)
    if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        target = target.with_suffix(".png")
    requested = (backend or "auto").lower()
    configured = (ctx.config.image_backend or "auto").lower()
    order: list[str]
    if requested != "auto":
        order = [requested]
    elif configured != "auto":
        order = [configured, "procedural"]
    elif ctx.config.allow_network:
        order = ["sd", "pollinations", "procedural"]
    else:
        order = ["procedural"]

    errors: list[str] = []
    image = None
    used = "procedural"
    for candidate in order:
        try:
            if candidate == "sd":
                image = render_sd(prompt, width, height, seed)
            elif candidate == "pollinations":
                image = render_pollinations(prompt, width, height, seed)
            else:
                candidate = "procedural"
                image = render_procedural(prompt, width, height, seed, style, light=bool(light))
            used = candidate
            break
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
    if image is None:
        raise ToolError("every image backend failed: " + " | ".join(errors))

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(target), format="PNG")
    return {
        "path": ctx.rel(target),
        "backend": used,
        "width": image.width,
        "height": image.height,
        "seed": int(seed),
        "style": style,
        "light": bool(light),
        "attempts": errors,
        "_artifacts": [
            artifact(
                ctx, target, "image", MIME,
                width=image.width, height=image.height, backend=used, seed=int(seed),
                light=bool(light), prompt=prompt[:300],
            )
        ],
    }


def register(registry) -> None:
    @registry.tool(
        "image.generate",
        "Generate a PNG image from a text prompt (local procedural by default; sd/pollinations optional).",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 720},
                "seed": {"type": "integer", "default": 0},
                "style": {"type": "string", "enum": list(STYLE_TUNING)},
                "backend": {"type": "string", "enum": ["auto", "procedural", "sd", "pollinations"]},
                "light": {"type": "boolean", "default": True,
                          "description": "light pastel art (default) or opt-in dark art"},
            },
            "required": ["path", "prompt"],
        },
        category="image",
        requires=["pillow", "numpy"],
    )
    def image_generate(
        ctx: ToolContext,
        path: str,
        prompt: str,
        width: int = 1280,
        height: int = 720,
        seed: int = 0,
        style: str = "editorial",
        backend: str = "auto",
        light: bool = True,
    ) -> ToolResult:
        data = generate(ctx, path, prompt, width, height, seed, style, backend, light)
        arts = data.pop("_artifacts", [])
        return ToolResult.success(
            data, [a if isinstance(a, Artifact) else Artifact(**a) for a in arts]
        )

    @registry.tool(
        "image.variations",
        "Generate a small set of image variants for one prompt and return their paths.",
        schema={
            "type": "object",
            "properties": {
                "path_prefix": {"type": "string"},
                "prompt": {"type": "string"},
                "count": {"type": "integer", "default": 3},
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 720},
                "style": {"type": "string"},
            },
            "required": ["path_prefix", "prompt"],
        },
        category="image",
        requires=["pillow", "numpy"],
    )
    def image_variations(
        ctx: ToolContext,
        path_prefix: str,
        prompt: str,
        count: int = 3,
        width: int = 1280,
        height: int = 720,
        style: str = "editorial",
    ) -> ToolResult:
        results, artifacts, errors = [], [], []
        for index in range(max(1, min(int(count), 8))):
            try:
                data = generate(
                    ctx, f"{path_prefix}-{index + 1}.png", prompt, width, height, index, style
                )
                arts = data.pop("_artifacts", [])
                results.append(data)
                artifacts.extend(arts)
            except Exception as exc:
                errors.append(f"variant {index + 1}: {exc}")
        if not results:
            return ToolResult.failure("no variants could be generated: " + " | ".join(errors))
        return ToolResult.success(
            {"count": len(results), "images": results, "errors": errors}, artifacts
        )
