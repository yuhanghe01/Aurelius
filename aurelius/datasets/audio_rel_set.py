"""
AudioRelSet: corpus of pairwise audio relations for relation-aware
text-to-audio generation.

An audio relation captures how two audio events are related, covering
three broad axes:

* **Temporal** – when one event occurs relative to the other:
  ``"before"``, ``"after"``, ``"simultaneous"``
* **Spatial** – where one event is positioned relative to the other:
  ``"left"``, ``"right"``, ``"center"``, ``"above"``, ``"below"``
* **Semantic** – how the two events are semantically connected:
  ``"similar"``, ``"contrasting"``, ``"complementary"``

Manifest format (JSON)
----------------------
[
    {
        "event_a": "dog barking",
        "event_b": "cat meowing",
        "relation_type": "temporal",
        "relation": "before",
        "description": "A dog barks before a cat meows.",   # optional
        "audio_path_a": "path/to/dog.wav",                  # optional
        "audio_path_b": "path/to/cat.wav"                   # optional
    },
    ...
]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Valid relation vocabulary
# ---------------------------------------------------------------------------

TEMPORAL_RELATIONS: Tuple[str, ...] = ("before", "after", "simultaneous")
SPATIAL_RELATIONS: Tuple[str, ...] = ("left", "right", "center", "above", "below")
SEMANTIC_RELATIONS: Tuple[str, ...] = ("similar", "contrasting", "complementary")

VALID_RELATION_TYPES: Tuple[str, ...] = ("temporal", "spatial", "semantic")
ALL_RELATIONS: Tuple[str, ...] = (
    TEMPORAL_RELATIONS + SPATIAL_RELATIONS + SEMANTIC_RELATIONS
)


@dataclass
class AudioRelation:
    """Metadata for a single pairwise audio relation entry.

    Attributes:
        event_a: Label of the first audio event.
        event_b: Label of the second audio event.
        relation_type: Category of the relation – one of
            ``"temporal"``, ``"spatial"``, or ``"semantic"``.
        relation: Specific relation value (e.g. ``"before"``).
        description: Optional free-text sentence describing the relation.
        audio_path_a: Optional path to the audio file for *event_a*.
        audio_path_b: Optional path to the audio file for *event_b*.
        extra: Additional arbitrary metadata.
    """

    event_a: str
    event_b: str
    relation_type: str
    relation: str
    description: Optional[str] = None
    audio_path_a: Optional[str] = None
    audio_path_b: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relation_type not in VALID_RELATION_TYPES:
            raise ValueError(
                f"relation_type must be one of {VALID_RELATION_TYPES}, "
                f"got {self.relation_type!r}."
            )
        valid_values = _relation_values_for_type(self.relation_type)
        if self.relation not in valid_values:
            raise ValueError(
                f"For relation_type={self.relation_type!r}, relation must be "
                f"one of {valid_values}, got {self.relation!r}."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioRelation":
        """Construct an :class:`AudioRelation` from a dictionary."""
        return cls(
            event_a=data["event_a"],
            event_b=data["event_b"],
            relation_type=data["relation_type"],
            relation=data["relation"],
            description=data.get("description"),
            audio_path_a=data.get("audio_path_a"),
            audio_path_b=data.get("audio_path_b"),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "event_a",
                    "event_b",
                    "relation_type",
                    "relation",
                    "description",
                    "audio_path_a",
                    "audio_path_b",
                }
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary (suitable for JSON output)."""
        d: Dict[str, Any] = {
            "event_a": self.event_a,
            "event_b": self.event_b,
            "relation_type": self.relation_type,
            "relation": self.relation,
        }
        if self.description is not None:
            d["description"] = self.description
        if self.audio_path_a is not None:
            d["audio_path_a"] = self.audio_path_a
        if self.audio_path_b is not None:
            d["audio_path_b"] = self.audio_path_b
        d.update(self.extra)
        return d

    @property
    def text_prompt(self) -> str:
        """Return a natural-language sentence describing this relation.

        If a *description* was provided it is returned as-is; otherwise a
        generic sentence is generated from the relation components.
        """
        if self.description:
            return self.description
        if self.relation_type == "temporal":
            if self.relation == "simultaneous":
                return f"{self.event_a} and {self.event_b} occur simultaneously."
            return f"{self.event_a} occurs {self.relation} {self.event_b}."
        if self.relation_type == "spatial":
            return (
                f"{self.event_a} is positioned to the {self.relation} "
                f"of {self.event_b}."
            )
        # semantic
        return f"{self.event_a} and {self.event_b} are {self.relation}."


def _relation_values_for_type(relation_type: str) -> Tuple[str, ...]:
    """Return the valid relation values for *relation_type*."""
    if relation_type == "temporal":
        return TEMPORAL_RELATIONS
    if relation_type == "spatial":
        return SPATIAL_RELATIONS
    return SEMANTIC_RELATIONS


class AudioRelSet(Dataset):
    """Corpus of pairwise audio relations.

    AudioRelSet stores a flat list of :class:`AudioRelation` entries and
    exposes a PyTorch ``Dataset`` interface.

    ``__getitem__`` returns a dict with the following keys:

    * ``"event_a"`` – label of the first event (str).
    * ``"event_b"`` – label of the second event (str).
    * ``"relation_type"`` – category of the relation (str).
    * ``"relation"`` – specific relation value (str).
    * ``"text_prompt"`` – natural-language description (str).
    * ``"relation_obj"`` – the raw :class:`AudioRelation` object.
    * ``"waveform_a"`` / ``"waveform_b"`` – waveform tensors, present
      only when the corresponding audio paths are available.

    Args:
        relations: List of :class:`AudioRelation` objects.
        sample_rate: Target sample rate for waveform loading.
        max_duration: If set, waveforms are trimmed/padded to this many
            seconds.
        transform: Optional callable applied to each loaded waveform.
    """

    def __init__(
        self,
        relations: List[AudioRelation],
        sample_rate: int = 22050,
        max_duration: Optional[float] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self._relations = list(relations)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.transform = transform

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        sample_rate: int = 22050,
        max_duration: Optional[float] = None,
        transform: Optional[Any] = None,
    ) -> "AudioRelSet":
        """Build an :class:`AudioRelSet` from a JSON manifest file.

        Args:
            manifest_path: Path to a JSON file containing a list of
                relation dictionaries (see module docstring for schema).
            sample_rate: Target sample rate for audio loading.
            max_duration: Optional maximum duration in seconds.
            transform: Optional waveform transform callable.

        Returns:
            A populated :class:`AudioRelSet` instance.
        """
        with open(manifest_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            raise ValueError(
                f"Manifest {manifest_path!r} must contain a JSON array."
            )
        relations = [AudioRelation.from_dict(r) for r in records]
        return cls(
            relations=relations,
            sample_rate=sample_rate,
            max_duration=max_duration,
            transform=transform,
        )

    @classmethod
    def from_list(
        cls,
        records: List[Dict[str, Any]],
        sample_rate: int = 22050,
        max_duration: Optional[float] = None,
        transform: Optional[Any] = None,
    ) -> "AudioRelSet":
        """Build an :class:`AudioRelSet` from a list of dicts.

        Args:
            records: List of relation dictionaries.
            sample_rate: Target sample rate for audio loading.
            max_duration: Optional maximum duration in seconds.
            transform: Optional waveform transform callable.

        Returns:
            A populated :class:`AudioRelSet` instance.
        """
        relations = [AudioRelation.from_dict(r) for r in records]
        return cls(
            relations=relations,
            sample_rate=sample_rate,
            max_duration=max_duration,
            transform=transform,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._relations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel = self._relations[idx]
        item: Dict[str, Any] = {
            "event_a": rel.event_a,
            "event_b": rel.event_b,
            "relation_type": rel.relation_type,
            "relation": rel.relation,
            "text_prompt": rel.text_prompt,
            "relation_obj": rel,
        }
        if rel.audio_path_a is not None:
            waveform_a, _ = self._load_waveform(rel.audio_path_a)
            if self.transform is not None:
                waveform_a = self.transform(waveform_a)
            item["waveform_a"] = waveform_a
        if rel.audio_path_b is not None:
            waveform_b, _ = self._load_waveform(rel.audio_path_b)
            if self.transform is not None:
                waveform_b = self.transform(waveform_b)
            item["waveform_b"] = waveform_b
        return item

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def relations(self) -> List[AudioRelation]:
        """Return the list of :class:`AudioRelation` entries (read-only view)."""
        return list(self._relations)

    @property
    def relation_types(self) -> List[str]:
        """Return the unique relation types present in the corpus."""
        return sorted({r.relation_type for r in self._relations})

    @property
    def text_prompts(self) -> List[str]:
        """Return the text prompt for every entry in corpus order."""
        return [r.text_prompt for r in self._relations]

    def filter_by_relation_type(self, relation_type: str) -> "AudioRelSet":
        """Return a new :class:`AudioRelSet` with only entries of the
        given *relation_type*.

        Args:
            relation_type: One of ``"temporal"``, ``"spatial"``,
                ``"semantic"``.

        Returns:
            Filtered :class:`AudioRelSet`.
        """
        if relation_type not in VALID_RELATION_TYPES:
            raise ValueError(
                f"relation_type must be one of {VALID_RELATION_TYPES}, "
                f"got {relation_type!r}."
            )
        filtered = [r for r in self._relations if r.relation_type == relation_type]
        return AudioRelSet(
            relations=filtered,
            sample_rate=self.sample_rate,
            max_duration=self.max_duration,
            transform=self.transform,
        )

    def filter_by_relation(self, relation: str) -> "AudioRelSet":
        """Return a new :class:`AudioRelSet` with only entries whose
        *relation* field equals *relation*.

        Args:
            relation: Relation value to filter on (e.g. ``"before"``).

        Returns:
            Filtered :class:`AudioRelSet`.
        """
        filtered = [r for r in self._relations if r.relation == relation]
        return AudioRelSet(
            relations=filtered,
            sample_rate=self.sample_rate,
            max_duration=self.max_duration,
            transform=self.transform,
        )

    def save_manifest(self, path: str) -> None:
        """Serialise the corpus to a JSON manifest file.

        Args:
            path: Output file path.
        """
        records = [r.to_dict() for r in self._relations]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio from *audio_path* and apply duration constraints."""
        from aurelius.utils.audio_utils import load_audio, pad_or_trim

        waveform, sr = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            mono=True,
            duration=self.max_duration,
        )
        if self.max_duration is not None:
            target_samples = int(self.max_duration * self.sample_rate)
            waveform = pad_or_trim(waveform, target_samples)
        return waveform, sr

    def __repr__(self) -> str:
        return (
            f"AudioRelSet("
            f"num_relations={len(self)}, "
            f"sample_rate={self.sample_rate}, "
            f"max_duration={self.max_duration})"
        )
