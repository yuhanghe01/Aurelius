# Aurelius

**Relation-Aware Text-to-Audio Generation** — ICLR 2026

Aurelius is a research framework for generating audio that faithfully
reflects not only the *content* of a text prompt but also the explicit
*relations* between audio events described in that prompt (e.g. temporal
ordering, spatial placement, and semantic relationships).

---

## Overview

| Module | Description |
|---|---|
| `AudioEventSet` | Corpus of individual audio events with labels and metadata |
| `AudioRelSet` | Corpus of pairwise audio relations (temporal / spatial / semantic) |
| `Aurelius` | Relation-aware latent diffusion model for text-to-audio generation |

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## AudioEventSet

`AudioEventSet` is a PyTorch `Dataset` that stores a flat list of
audio events. Each event has a short textual label (e.g. `"dog barking"`),
an optional audio file path, an optional semantic category, and free-form
tags.

### Manifest format (JSON)

```json
[
  {
    "label": "dog barking",
    "audio_path": "data/dog_bark.wav",
    "category": "animal",
    "duration": 3.5,
    "tags": ["dog", "bark"]
  },
  {
    "label": "car horn",
    "category": "vehicle",
    "duration": 1.2
  }
]
```

### Usage

```python
from aurelius import AudioEventSet

# Build from a JSON manifest
dataset = AudioEventSet.from_manifest("events.json", sample_rate=22050)

# Build directly from a list of dicts
dataset = AudioEventSet.from_list([
    {"label": "dog barking", "category": "animal"},
    {"label": "car horn",    "category": "vehicle"},
])

print(len(dataset))         # 2
print(dataset.labels)       # ['dog barking', 'car horn']
print(dataset.categories)   # ['animal', 'vehicle']

# Filter
animals = dataset.filter_by_category("animal")

# PyTorch DataLoader compatible
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16)

# Save back to disk
dataset.save_manifest("events_out.json")
```

---

## AudioRelSet

`AudioRelSet` is a PyTorch `Dataset` that stores pairwise audio
relations. Each entry links two events with a typed relation.

### Relation types and values

| Type | Values |
|---|---|
| `temporal` | `before`, `after`, `simultaneous` |
| `spatial` | `left`, `right`, `center`, `above`, `below` |
| `semantic` | `similar`, `contrasting`, `complementary` |

### Manifest format (JSON)

```json
[
  {
    "event_a": "dog barking",
    "event_b": "cat meowing",
    "relation_type": "temporal",
    "relation": "before",
    "description": "A dog barks before a cat meows.",
    "audio_path_a": "data/dog.wav",
    "audio_path_b": "data/cat.wav"
  },
  {
    "event_a": "thunder",
    "event_b": "rain",
    "relation_type": "spatial",
    "relation": "left"
  }
]
```

### Usage

```python
from aurelius import AudioRelSet

# Build from a JSON manifest
dataset = AudioRelSet.from_manifest("relations.json")

# Build directly from a list of dicts
dataset = AudioRelSet.from_list([
    {
        "event_a": "dog barking", "event_b": "cat meowing",
        "relation_type": "temporal", "relation": "before",
    },
    {
        "event_a": "thunder", "event_b": "rain",
        "relation_type": "spatial", "relation": "left",
    },
])

print(len(dataset))           # 2
print(dataset.relation_types) # ['spatial', 'temporal']
print(dataset.text_prompts)
# ['A dog barking occurs before a cat meowing.',
#  'thunder is positioned to the left of rain.']

# Filter
temporal = dataset.filter_by_relation_type("temporal")
before   = dataset.filter_by_relation("before")

# Each item exposes a text_prompt for use with the model
item = dataset[0]
print(item["text_prompt"])  # "A dog barking occurs before a cat meowing."

# Save back to disk
dataset.save_manifest("relations_out.json")
```

---

## Aurelius Model

`Aurelius` is a latent diffusion model conditioned on both a text
embedding and an explicit relation encoding.

```
text prompt  ──► TextEncoder         ──► text_emb
relation     ──► RelationEncoder     ──► rel_emb
                 ConditioningProjector ──► cond_emb
noise        ──► UNet1D (diffusion)   ──► latent
                 AudioDecoder         ──► waveform
```

### Training

```python
import torch
from aurelius import Aurelius

model = Aurelius(
    text_dim=512,
    rel_embed_dim=256,
    cond_dim=512,
    latent_channels=8,
    latent_length=256,
    num_diffusion_steps=1000,
)

# Encode text with the built-in stub encoder (or supply your own)
texts = ["A dog barks before a cat meows."]
text_emb = model.encode_text(texts)           # (B, text_dim)

# Provide ground-truth latents to compute the DDPM loss
target_latents = torch.randn(1, 8, 256)
output = model(
    text_emb,
    relation_types=["temporal"],
    relations=["before"],
    target_latents=target_latents,
)
loss = output["loss"]
loss.backward()
```

### Inference

```python
latents = model.generate(
    text_emb,
    relation_types=["temporal"],
    relations=["before"],
    num_steps=50,
)
# latents: (B, latent_channels, latent_length)
# Pass through a vocoder to obtain a waveform.
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Project structure

```
aurelius/
├── datasets/
│   ├── audio_event_set.py   # AudioEventSet corpus
│   └── audio_rel_set.py     # AudioRelSet corpus
├── models/
│   └── aurelius.py          # Aurelius diffusion model
└── utils/
    └── audio_utils.py       # Audio I/O helpers
tests/
├── test_audio_event_set.py
├── test_audio_rel_set.py
└── test_aurelius_model.py
```
