"""Shared harness for the EPYC 7302 TTS throughput experiments.

Provides: corpus of sentences, utterance text assembly, model loading,
per-utterance timing, CPU affinity helpers, minimal WAV writer, JSON results.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("/tmp/model-local")
VOICES = {
    "a": "/tmp/voice-out/anna.safetensors",
    "b": "/tmp/voice-out/charles.safetensors",
    "c": "/tmp/voice-out/eve.safetensors",
    "d": "/tmp/voice-out/fantine.safetensors",
}
SAMPLE_RATE = 24000

# Public-domain style sentences (pooled so every utterance differs).
SENTENCES = [
    "The quick brown fox jumps over the lazy dog, and the dog sleeps peacefully by the fire.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "She walked through the garden gate and into a world of green hedges and quiet paths.",
    "The old lighthouse stood alone on the cliff, its beam sweeping the dark water every night.",
    "He opened the heavy book and the smell of old paper filled the small room.",
    "The train rattled through the countryside, past fields of golden wheat and red barns.",
    "A gentle rain began to fall, tapping softly on the windowpanes like a distant melody.",
    "The baker rose before dawn to light the ovens and knead the day's first loaves.",
    "Mountains rose in the distance, their peaks wrapped in clouds that never seemed to move.",
    "She hummed a tune she had heard on the radio and tapped her foot in time with the beat.",
    "The library was quiet except for the rustle of pages and the ticking of a grandfather clock.",
    "He counted the coins on the table twice, then pushed them into a neat little stack.",
    "The harbor was busy that morning with fishing boats returning from the open sea.",
    "Autumn leaves drifted across the lawn, red and gold against the gray afternoon sky.",
    "The cat stretched lazily on the windowsill and watched the sparrows in the cherry tree.",
    "They built a fire on the beach and told stories until the stars came out one by one.",
    "The museum's new wing displayed paintings from the Renaissance and early baroque periods.",
    "She wrote a long letter to her grandmother describing the city and all its wonders.",
    "The wind howled around the cabin while the stove glowed and the kettle began to sing.",
    "He studied the map for a long time, tracing the river with his finger and planning the route.",
    "The children ran through the meadow chasing butterflies and shouting with delight.",
    "A cold mist settled over the valley, hiding the river and muffling every sound.",
    "The clock tower chimed six times, and the streets began to fill with evening traffic.",
    "She practiced the piano for an hour, repeating the difficult passage until her fingers ached.",
    "The farmer checked the fence posts along the north field, hammering loose nails as he went.",
    "Rainwater collected in the barrels beside the greenhouse, ready for the dry weeks ahead.",
    "The astronomer adjusted the telescope and focused on the faint glow of a distant nebula.",
    "They rowed across the lake in silence, the oars dipping and rising in a steady rhythm.",
    "The village square was decorated with lanterns and garlands for the harvest festival.",
    "He read the newspaper over breakfast, turning the pages slowly and sipping his coffee.",
    "The dog barked once at the mailman, then wagged its tail and settled back on the porch.",
    "She arranged the wildflowers in a blue vase and placed them on the kitchen windowsill.",
    "The bridge spanned the gorge, its steel cables gleaming in the afternoon sun.",
    "They hiked for hours along the ridge, stopping now and then to admire the view.",
    "The chemist mixed the solutions carefully, noting the color change in her logbook.",
    "A flock of geese flew overhead in a loose formation, calling to one another as they passed.",
    "The carpenter measured twice and cut once, following the chalk lines on the oak board.",
    "She remembered the summers of her childhood, long afternoons by the lake with her cousins.",
    "The theatre lights dimmed and the audience grew quiet as the curtain began to rise.",
    "He repaired the old bicycle in the garage, oiling the chain and adjusting the brakes.",
    "The market was crowded with stalls selling fruit, flowers, cheese, and warm bread.",
    "Snow covered the rooftops overnight, and the whole town woke to a soft white silence.",
    "The professor explained the theory with diagrams on the blackboard, answering every question.",
    "They sailed south along the coast, past cliffs and coves, keeping the lighthouse in sight.",
    "The kitten batted at the ball of yarn until it unraveled across the entire living room.",
    "He wrote his name carefully at the top of the page and began the examination.",
    "The river curved through the meadow, glinting between the willows in the evening light.",
    "She baked a loaf of bread and the kitchen filled with the warm scent of yeast and butter.",
    "The soldier stood at attention while the flag was raised, the morning bugle echoing.",
    "They watched the thunderstorm roll in from the west, counting the seconds between flash and boom.",
    "The gardener pruned the roses and tied the new shoots to the wooden trellis.",
    "He walked the dog through the park every evening, rain or shine, without fail.",
    "The orchestra tuned its instruments, a soft wave of sound washing over the hall.",
    "She read the recipe twice, measured the flour, and cracked the eggs into the bowl.",
    "The train arrived at the platform exactly on time, its doors opening with a hiss.",
    "They camped under the pines and listened to the wind moving through the branches.",
    "The detective examined the room, noting the open window and the chair pushed aside.",
    "A thick fog rolled in from the sea, wrapping the town in a damp gray blanket.",
    "The students gathered in the courtyard, chatting and laughing before the first bell.",
    "He played chess with his grandfather every Sunday, losing more often than he won.",
    "The fountain in the square sent water arcing into the air, catching the sunlight.",
    "She knitted a scarf through the winter evenings, row after row of soft blue wool.",
    "The harvest moon rose over the fields, round and bright, as the workers finished the day.",
    "They climbed the tower stairs, counting the steps, and emerged onto a windy balcony.",
    "The librarian stamped the due date into the books and slid them across the counter.",
    "He filled the bird feeder with sunflower seeds and hung it from the maple branch.",
    "The canyon walls rose sheer on both sides, striped with red rock and shadow.",
    "She learned to sail that summer, mastering the ropes and reading the wind.",
    "The bakery opened early, its display cases gleaming with pastries and fresh tarts.",
    "They sat on the porch swing, talking quietly while the fireflies blinked in the dusk.",
    "The glacier crept down the valley, grinding the rock beneath it into fine gray dust.",
    "He tuned the guitar and played a slow song, the notes floating out across the water.",
    "The postal worker sorted the letters by neighborhood, whistling as she worked.",
    "A heron stood motionless in the shallows, watching for fish with unblinking patience.",
    "The old mill wheel turned slowly in the stream, water sparkling over the worn wood.",
    "She sketched the harbor scene in charcoal, shading the boats and the masts.",
    "The radio crackled with the weather report, warning of storms moving in from the coast.",
    "They planted a row of sunflowers along the fence, and watered them every morning.",
    "The blacksmith shaped the glowing iron on the anvil, sparks flying with each strike.",
    "He memorized the poem line by line, reciting it softly to himself as he walked.",
    "The ferry crossed the bay at dawn, gulls following in its wake.",
    "She folded the letter carefully and sealed it with a drop of red wax.",
    "The canyon echoed with the sound of the river far below, a constant low thunder.",
    "They played cards by the fire until the candles burned down to their holders.",
    "The museum guard nodded as the last visitors left, then locked the heavy doors.",
    "He planted the sapling in the yard, patting the soil around its roots.",
    "The tide came in slowly, covering the sandbars and rocking the moored boats.",
    "She took the scenic route home, driving past orchards and stone walls.",
    "The choir practiced the hymn again and again until the harmonies were perfect.",
    "They watched the sunrise from the hilltop, the valley filling with golden light.",
    "The engineer inspected the bridge girders, marking the rivets that needed attention.",
    "He wrote postcards to his friends, describing the markets and the narrow streets.",
    "The rabbit dashed across the garden and disappeared under the hedge.",
    "She measured the curtains against the window, then pinned the fabric in place.",
    "The orchestra conductor raised his baton, and the hall fell into expectant silence.",
    "They gathered driftwood for the bonfire as the sun dipped below the horizon.",
    "The scholar pored over the manuscript, deciphering the faded handwriting.",
    "A hawk circled high above the field, riding the warm air in slow spirals.",
    "He cleaned the lenses of his spectacles and returned to the crossword puzzle.",
    "The waterfall thundered into the pool below, sending up a cool mist.",
    "She planted tulip bulbs in the autumn, dreaming of the spring bloom.",
    "The courier cycled through the old town, dodging cobblestones and pigeons.",
    "They tasted the wine at the vineyard, comparing the vintages with serious faces.",
    "The fireflies rose from the meadow as dusk fell, a thousand tiny lanterns.",
    "He practiced his speech in front of the mirror, gesturing with both hands.",
    "The snowplow cleared the roads before sunrise, its lights flashing in the dark.",
    "She found a seashell on the beach and held it to her ear, hearing the ocean.",
    "The historian examined the old photographs, identifying faces and places.",
    "They paddled the canoe up the quiet stream, past overhanging branches.",
    "The owl called from the barn roof as the last light faded from the sky.",
    "He sharpened the pencils and laid out the blank pages for the morning's work.",
    "The dancer rehearsed the same sequence of steps until they became effortless.",
    "She watered the plants on the balcony and watched the city lights come on.",
    "The shepherd moved the flock to higher pastures as the weather turned warm.",
    "They celebrated the harvest with a long table under the trees and music late into the night.",
]


def build_utterances(n: int, target_chars: int = 660, seed: int = 7) -> list[str]:
    """Assemble `n` distinct utterances, each ~target_chars of prose."""
    rng = np.random.default_rng(seed)
    sents = [s for s in SENTENCES]
    rng.shuffle(sents)
    utterances = []
    idx = 0
    for _ in range(n):
        parts = []
        total = 0
        while total < target_chars - 60:
            s = sents[idx % len(sents)]
            idx += 1
            parts.append(s)
            total += len(s) + 1
        utterances.append(" ".join(parts))
    return utterances


def load_model(quant: str | None, temperature: float = 0.5):
    import ptts
    if quant in (None, "None", "f32", "none"):
        quant = None
    model = ptts.load_model(temperature=temperature, config=str(MODEL_DIR / "config.json"), quant=quant)
    for name, path in VOICES.items():
        model.add_voice(name, path)
    return model


def pin_current(cpu: int):
    """Pin the calling thread to one logical CPU. Returns previous mask."""
    import os
    prev = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cpu})
    return prev


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE):
    pcm = np.asarray(pcm, dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2")
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write((36 + data.nbytes).to_bytes(4, "little"))
        f.write(b"WAVEfmt ")
        f.write((16).to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little"))
        f.write((1).to_bytes(2, "little"))
        f.write(sample_rate.to_bytes(4, "little"))
        f.write((sample_rate * 2).to_bytes(4, "little"))
        f.write((2).to_bytes(2, "little"))
        f.write((16).to_bytes(2, "little"))
        f.write(b"data")
        f.write(data.nbytes.to_bytes(4, "little"))
        f.write(data.tobytes())


def save_result(name: str, payload: dict):
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def fmt_rtf(rtf: float) -> str:
    return f"{rtf:.3f}x"


def measure_rtf(state, text: str, temperature: float = 0.5, seed: int = 42) -> dict:
    """Generate one utterance; return timing + audio stats."""
    import time
    t0 = time.perf_counter()
    pcm = state.generate_audio(text, temperature=temperature, seed=seed)
    dt = time.perf_counter() - t0
    pcm = np.asarray(pcm, dtype=np.float32)
    dur = pcm.shape[0] / SAMPLE_RATE
    return {
        "wall_s": dt,
        "audio_s": dur,
        "rtf": dt / dur if dur > 0 else float("inf"),
        "samples": int(pcm.shape[0]),
        "rms": float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0,
        "nan": bool(np.isnan(pcm).any()),
    }
