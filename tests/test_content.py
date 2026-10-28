"""Request interpretation: intents, topic extraction and structured plans."""

from quanai.content import (
    app_plan,
    deck_plan,
    detect_domain,
    detect_framework,
    detect_intents,
    image_prompts,
    titlecase,
    topic_of,
    video_plan,
)


def test_detects_deck_intent():
    assert "deck" in detect_intents("build me a powerpoint deck about climate policy")


def test_detects_multi_artifact_request():
    intents = detect_intents("make a deck, an app and a promo video about solar power")
    assert {"deck", "video"} <= set(intents)
    assert {"desktop", "mobile"} & set(intents)


def test_detects_named_framework():
    assert detect_framework("build a react native app") == ("react-native", "mobile")
    assert detect_framework("an electron desktop tool") == ("electron", "desktop")
    assert detect_framework("something nice") == (None, None)


def test_media_analysis_request_does_not_trigger_video_generation():
    assert detect_intents("analyse this video clip and report on its structure") == ["media_in"]


def test_generation_verb_keeps_video_intent():
    assert "video" in detect_intents("make a promo video from clip.mp4")


def test_topic_and_domain_extraction():
    topic = topic_of("Create a 7 slide powerpoint deck about solar energy adoption")
    assert "solar energy" in topic
    assert "slide" not in topic and "deck" not in topic
    assert topic_of("generate an abstract poster image about deep ocean exploration") == (
        "deep ocean exploration"
    )
    assert detect_domain("a deck about AI models and data") == "technology"
    assert titlecase("the state of ai in schools") == "The State of AI in Schools"


def test_deck_plan_structure():
    plan = deck_plan("a presentation about urban cycling", 6)
    assert len(plan.slides) == 6
    assert plan.slides[0].kind == "title"
    assert plan.slides[-1].kind == "closing"
    assert all(s.notes for s in plan.slides)


def test_video_plan_scene_count_and_duration():
    plan = video_plan("a promo about our agent", 4, 3.0)
    assert len(plan["scenes"]) == 4
    assert plan["duration"] == 12.0
    assert all(scene["caption"] for scene in plan["scenes"])


def test_app_plan_defaults_per_platform():
    desktop = app_plan("a habit tracker", "desktop")
    mobile = app_plan("a habit tracker", "mobile")
    assert desktop["framework"] == "tkinter"
    assert mobile["framework"] == "pwa"
    assert desktop["features"] and desktop["name"].isalnum()


def test_image_prompts_are_varied_and_deterministic():
    first = image_prompts("deep ocean poster", 3)
    second = image_prompts("deep ocean poster", 3)
    assert first == second
    assert len({p["prompt"] for p in first}) == 3
    assert all("deep ocean" in p["prompt"] for p in first)


def test_requested_slides_reads_the_number():
    from quanai.content import requested_slides

    assert requested_slides("create a 7 slide deck about solar") == 7
    assert requested_slides("a 12-slide presentation") == 12
    assert requested_slides("a deck about solar") == 8
    assert requested_slides("a 2 slide deck") == 4  # clamped to a usable minimum
    assert requested_slides("a 40 slide deck") == 24  # clamped to a sane maximum


def test_bullets_come_from_the_subject_not_the_format_words():
    plan = deck_plan("create a 7 slide powerpoint deck about solar energy adoption", 7)
    body = " ".join(bullet for slide in plan.slides for bullet in slide.bullets).lower()
    assert "solar energy" in body
    assert "slide" not in body and "powerpoint" not in body and "deck" not in body
