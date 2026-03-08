"""
Tests for AudioEventSet.
"""

import json
import os
import tempfile
import unittest

import torch

from aurelius.datasets.audio_event_set import AudioEvent, AudioEventSet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "label": "dog barking",
        "category": "animal",
        "duration": 3.0,
        "tags": ["dog", "bark"],
    },
    {
        "label": "car horn",
        "category": "vehicle",
        "duration": 1.5,
        "tags": ["car", "horn"],
    },
    {
        "label": "glass breaking",
        "category": "impact",
        "duration": 0.8,
        "tags": ["glass", "break", "impact"],
    },
    {
        "label": "cat meowing",
        "category": "animal",
        "duration": 2.0,
        "tags": ["cat", "meow"],
    },
]


# ---------------------------------------------------------------------------
# AudioEvent unit tests
# ---------------------------------------------------------------------------


class TestAudioEvent(unittest.TestCase):
    def test_from_dict_basic(self):
        event = AudioEvent.from_dict({"label": "dog barking", "category": "animal"})
        self.assertEqual(event.label, "dog barking")
        self.assertEqual(event.category, "animal")
        self.assertIsNone(event.audio_path)
        self.assertIsNone(event.duration)
        self.assertEqual(event.tags, [])

    def test_from_dict_full(self):
        event = AudioEvent.from_dict(SAMPLE_RECORDS[0])
        self.assertEqual(event.label, "dog barking")
        self.assertEqual(event.category, "animal")
        self.assertAlmostEqual(event.duration, 3.0)
        self.assertEqual(event.tags, ["dog", "bark"])

    def test_to_dict_roundtrip(self):
        original = SAMPLE_RECORDS[1]
        event = AudioEvent.from_dict(original)
        d = event.to_dict()
        self.assertEqual(d["label"], original["label"])
        self.assertEqual(d["category"], original["category"])
        self.assertAlmostEqual(d["duration"], original["duration"])
        self.assertEqual(d["tags"], original["tags"])

    def test_extra_fields_preserved(self):
        data = {"label": "thunder", "source": "freesound", "id": 42}
        event = AudioEvent.from_dict(data)
        self.assertEqual(event.extra["source"], "freesound")
        self.assertEqual(event.extra["id"], 42)
        d = event.to_dict()
        self.assertEqual(d["source"], "freesound")
        self.assertEqual(d["id"], 42)

    def test_missing_label_raises(self):
        with self.assertRaises(KeyError):
            AudioEvent.from_dict({"category": "animal"})


# ---------------------------------------------------------------------------
# AudioEventSet unit tests
# ---------------------------------------------------------------------------


class TestAudioEventSet(unittest.TestCase):
    def _make_set(self, records=None):
        return AudioEventSet.from_list(records or SAMPLE_RECORDS)

    # ---- construction ----

    def test_from_list(self):
        ds = self._make_set()
        self.assertEqual(len(ds), 4)

    def test_from_manifest(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(SAMPLE_RECORDS, fh)
            path = fh.name
        try:
            ds = AudioEventSet.from_manifest(path)
            self.assertEqual(len(ds), 4)
        finally:
            os.unlink(path)

    def test_from_manifest_bad_json_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump({"not": "a list"}, fh)
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                AudioEventSet.from_manifest(path)
        finally:
            os.unlink(path)

    # ---- accessors ----

    def test_labels(self):
        ds = self._make_set()
        self.assertEqual(ds.labels, [r["label"] for r in SAMPLE_RECORDS])

    def test_categories(self):
        ds = self._make_set()
        cats = ds.categories
        self.assertIn("animal", cats)
        self.assertIn("vehicle", cats)
        self.assertIn("impact", cats)

    def test_events_property_returns_copy(self):
        ds = self._make_set()
        events_a = ds.events
        events_b = ds.events
        # Mutating the returned list must not affect the dataset
        events_a.append(AudioEvent(label="extra"))
        self.assertEqual(len(ds), 4)
        self.assertEqual(len(events_b), 4)

    # ---- filtering ----

    def test_filter_by_category(self):
        ds = self._make_set()
        animal_ds = ds.filter_by_category("animal")
        self.assertEqual(len(animal_ds), 2)
        for event in animal_ds.events:
            self.assertEqual(event.category, "animal")

    def test_filter_by_category_no_match(self):
        ds = self._make_set()
        result = ds.filter_by_category("robot")
        self.assertEqual(len(result), 0)

    def test_filter_by_label(self):
        ds = self._make_set()
        result = ds.filter_by_label("car horn")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.events[0].label, "car horn")

    # ---- __getitem__ (no audio paths) ----

    def test_getitem_no_audio_path(self):
        ds = self._make_set()
        item = ds[0]
        self.assertIn("label", item)
        self.assertIn("event", item)
        self.assertNotIn("waveform", item)
        self.assertNotIn("sample_rate", item)
        self.assertEqual(item["label"], "dog barking")

    # ---- save_manifest ----

    def test_save_manifest(self):
        ds = self._make_set()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            ds.save_manifest(path)
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(len(loaded), 4)
            self.assertEqual(loaded[0]["label"], "dog barking")
        finally:
            os.unlink(path)

    # ---- repr ----

    def test_repr(self):
        ds = self._make_set()
        r = repr(ds)
        self.assertIn("AudioEventSet", r)
        self.assertIn("4", r)

    # ---- default params ----

    def test_default_sample_rate(self):
        ds = AudioEventSet.from_list(SAMPLE_RECORDS)
        self.assertEqual(ds.sample_rate, 22050)

    def test_custom_sample_rate(self):
        ds = AudioEventSet.from_list(SAMPLE_RECORDS, sample_rate=16000)
        self.assertEqual(ds.sample_rate, 16000)

    # ---- transform ----

    def test_transform_not_applied_without_waveform(self):
        """Transform should not crash when no waveform is present."""
        called = []

        def my_transform(w):
            called.append(True)
            return w

        ds = AudioEventSet.from_list(SAMPLE_RECORDS, transform=my_transform)
        _ = ds[0]
        self.assertEqual(len(called), 0)


if __name__ == "__main__":
    unittest.main()
