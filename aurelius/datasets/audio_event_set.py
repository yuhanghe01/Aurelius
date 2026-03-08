"""
AudioEventSet: corpus of individual audio events for relation-aware
text-to-audio generation.

An audio event is a single, semantically coherent sound occurrence
(e.g., "dog barking", "car horn", "glass breaking").  AudioEventSet
provides a PyTorch-compatible Dataset interface and supports loading
from a JSON manifest that lists each event entry together with its
label and (optionally) an audio file path.

Manifest format (JSON)
----------------------
[
    {
        "audio_path": "path/to/audio.wav",   # optional
        "label": "dog barking",
        "category": "animal",                # optional
        "duration": 3.5,                     # seconds, optional
        "tags": ["dog", "bark"]              # optional
    },
    ...
]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class AudioEvent:
    """Metadata for a single audio event entry.

    Attributes:
        label: Short textual label for the event (e.g. "dog barking").
        audio_path: Path to the corresponding audio file, or None if
            the entry is label-only.
        category: High-level semantic category (e.g. "animal").
        duration: Duration of the audio event in seconds.
        tags: List of free-form keyword tags.
        extra: Additional arbitrary metadata.
    """

    label: str
    audio_path: Optional[str] = None
    category: Optional[str] = None
    duration: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioEvent":
        """Construct an :class:`AudioEvent` from a dictionary."""
        return cls(
            label=data["label"],
            audio_path=data.get("audio_path"),
            category=data.get("category"),
            duration=data.get("duration"),
            tags=data.get("tags", []),
            extra={
                k: v
                for k, v in data.items()
                if k not in {"label", "audio_path", "category", "duration", "tags"}
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary (suitable for JSON output)."""
        d: Dict[str, Any] = {"label": self.label}
        if self.audio_path is not None:
            d["audio_path"] = self.audio_path
        if self.category is not None:
            d["category"] = self.category
        if self.duration is not None:
            d["duration"] = self.duration
        if self.tags:
            d["tags"] = self.tags
        d.update(self.extra)
        return d


class AudioEventSet(Dataset):
    """Corpus of individual audio events.

    AudioEventSet stores a flat list of :class:`AudioEvent` entries and
    exposes a PyTorch ``Dataset`` interface so that it integrates
    seamlessly with ``DataLoader``.

    When audio paths are present, ``__getitem__`` returns a dict with
    keys ``"waveform"``, ``"sample_rate"``, ``"label"``, and
    ``"event"`` (the :class:`AudioEvent` object).  When an entry has no
    audio path the ``"waveform"`` and ``"sample_rate"`` keys are
    omitted.

    Args:
        events: List of :class:`AudioEvent` objects.
        sample_rate: Target sample rate used when loading waveforms.
        max_duration: If set, waveforms are trimmed/padded to this many
            seconds.
        transform: Optional callable applied to the waveform tensor
            after loading.
    """

    def __init__(
        self,
        events: List[AudioEvent],
        sample_rate: int = 22050,
        max_duration: Optional[float] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self._events = list(events)
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
    ) -> "AudioEventSet":
        """Build an :class:`AudioEventSet` from a JSON manifest file.

        Args:
            manifest_path: Path to a JSON file containing a list of
                event dictionaries (see module docstring for schema).
            sample_rate: Target sample rate for audio loading.
            max_duration: Optional maximum event duration in seconds.
            transform: Optional waveform transform callable.

        Returns:
            A populated :class:`AudioEventSet` instance.
        """
        with open(manifest_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            raise ValueError(
                f"Manifest {manifest_path!r} must contain a JSON array."
            )
        events = [AudioEvent.from_dict(r) for r in records]
        return cls(
            events=events,
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
    ) -> "AudioEventSet":
        """Build an :class:`AudioEventSet` from a list of dicts.

        Args:
            records: List of event dictionaries (see module docstring).
            sample_rate: Target sample rate for audio loading.
            max_duration: Optional maximum event duration in seconds.
            transform: Optional waveform transform callable.

        Returns:
            A populated :class:`AudioEventSet` instance.
        """
        events = [AudioEvent.from_dict(r) for r in records]
        return cls(
            events=events,
            sample_rate=sample_rate,
            max_duration=max_duration,
            transform=transform,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        event = self._events[idx]
        item: Dict[str, Any] = {"label": event.label, "event": event}
        if event.audio_path is not None:
            waveform, sr = self._load_waveform(event.audio_path)
            if self.transform is not None:
                waveform = self.transform(waveform)
            item["waveform"] = waveform
            item["sample_rate"] = sr
        return item

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def events(self) -> List[AudioEvent]:
        """Return the list of :class:`AudioEvent` entries (read-only view)."""
        return list(self._events)

    @property
    def labels(self) -> List[str]:
        """Return all event labels in corpus order."""
        return [e.label for e in self._events]

    @property
    def categories(self) -> List[str]:
        """Return the unique categories present in the corpus."""
        return sorted({e.category for e in self._events if e.category is not None})

    def filter_by_category(self, category: str) -> "AudioEventSet":
        """Return a new :class:`AudioEventSet` containing only events in
        the specified *category*.

        Args:
            category: Category string to filter on.

        Returns:
            A new :class:`AudioEventSet` with the filtered events.
        """
        filtered = [e for e in self._events if e.category == category]
        return AudioEventSet(
            events=filtered,
            sample_rate=self.sample_rate,
            max_duration=self.max_duration,
            transform=self.transform,
        )

    def filter_by_label(self, label: str) -> "AudioEventSet":
        """Return a new :class:`AudioEventSet` containing only events
        whose label matches *label* (exact match).

        Args:
            label: Label string to filter on.

        Returns:
            A new :class:`AudioEventSet` with the filtered events.
        """
        filtered = [e for e in self._events if e.label == label]
        return AudioEventSet(
            events=filtered,
            sample_rate=self.sample_rate,
            max_duration=self.max_duration,
            transform=self.transform,
        )

    def save_manifest(self, path: str) -> None:
        """Serialise the corpus to a JSON manifest file.

        Args:
            path: Output file path.
        """
        records = [e.to_dict() for e in self._events]
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
            f"AudioEventSet("
            f"num_events={len(self)}, "
            f"sample_rate={self.sample_rate}, "
            f"max_duration={self.max_duration})"
        )
