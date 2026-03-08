"""
Tests for AudioRelSet.
"""

import json
import os
import tempfile
import unittest

import torch

from aurelius.datasets.audio_rel_set import (
    ALL_RELATIONS,
    SEMANTIC_RELATIONS,
    SPATIAL_RELATIONS,
    TEMPORAL_RELATIONS,
    VALID_RELATION_TYPES,
    AudioRelation,
    AudioRelSet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "event_a": "dog barking",
        "event_b": "cat meowing",
        "relation_type": "temporal",
        "relation": "before",
        "description": "A dog barks before a cat meows.",
    },
    {
        "event_a": "car horn",
        "event_b": "ambulance siren",
        "relation_type": "temporal",
        "relation": "simultaneous",
    },
    {
        "event_a": "thunder",
        "event_b": "rain",
        "relation_type": "spatial",
        "relation": "left",
        "description": "Thunder is positioned to the left of rain.",
    },
    {
        "event_a": "piano melody",
        "event_b": "violin melody",
        "relation_type": "semantic",
        "relation": "complementary",
    },
]


# ---------------------------------------------------------------------------
# AudioRelation unit tests
# ---------------------------------------------------------------------------


class TestAudioRelation(unittest.TestCase):
    def test_from_dict_basic(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[0])
        self.assertEqual(rel.event_a, "dog barking")
        self.assertEqual(rel.event_b, "cat meowing")
        self.assertEqual(rel.relation_type, "temporal")
        self.assertEqual(rel.relation, "before")
        self.assertEqual(rel.description, "A dog barks before a cat meows.")

    def test_from_dict_no_description(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[1])
        self.assertIsNone(rel.description)

    def test_to_dict_roundtrip(self):
        original = SAMPLE_RECORDS[0]
        rel = AudioRelation.from_dict(original)
        d = rel.to_dict()
        self.assertEqual(d["event_a"], original["event_a"])
        self.assertEqual(d["event_b"], original["event_b"])
        self.assertEqual(d["relation_type"], original["relation_type"])
        self.assertEqual(d["relation"], original["relation"])
        self.assertEqual(d["description"], original["description"])

    def test_invalid_relation_type_raises(self):
        data = {**SAMPLE_RECORDS[0], "relation_type": "nonexistent"}
        with self.assertRaises(ValueError):
            AudioRelation.from_dict(data)

    def test_invalid_relation_value_raises(self):
        data = {**SAMPLE_RECORDS[0], "relation": "never"}
        with self.assertRaises(ValueError):
            AudioRelation.from_dict(data)

    def test_relation_value_wrong_type_raises(self):
        # 'left' is spatial, not temporal
        data = {**SAMPLE_RECORDS[0], "relation": "left"}
        with self.assertRaises(ValueError):
            AudioRelation.from_dict(data)

    # ---- text_prompt ----

    def test_text_prompt_uses_description_when_present(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[0])
        self.assertEqual(rel.text_prompt, "A dog barks before a cat meows.")

    def test_text_prompt_temporal_before(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[0])
        # description present, so use it
        self.assertIn("before", rel.text_prompt)

    def test_text_prompt_temporal_no_description(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[1])
        prompt = rel.text_prompt
        self.assertIn("car horn", prompt)
        self.assertIn("ambulance siren", prompt)
        self.assertIn("simultaneously", prompt)

    def test_text_prompt_spatial(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[2])
        # description present
        self.assertIn("left", rel.text_prompt)

    def test_text_prompt_spatial_no_description(self):
        data = {
            "event_a": "footsteps",
            "event_b": "music",
            "relation_type": "spatial",
            "relation": "right",
        }
        rel = AudioRelation.from_dict(data)
        prompt = rel.text_prompt
        self.assertIn("footsteps", prompt)
        self.assertIn("music", prompt)
        self.assertIn("right", prompt)

    def test_text_prompt_semantic_no_description(self):
        rel = AudioRelation.from_dict(SAMPLE_RECORDS[3])
        prompt = rel.text_prompt
        self.assertIn("piano melody", prompt)
        self.assertIn("violin melody", prompt)
        self.assertIn("complementary", prompt)

    # ---- extra fields ----

    def test_extra_fields_preserved(self):
        data = {**SAMPLE_RECORDS[0], "source": "manual", "split": "train"}
        rel = AudioRelation.from_dict(data)
        self.assertEqual(rel.extra["source"], "manual")
        self.assertEqual(rel.extra["split"], "train")

    def test_extra_fields_in_to_dict(self):
        data = {**SAMPLE_RECORDS[0], "score": 0.9}
        rel = AudioRelation.from_dict(data)
        d = rel.to_dict()
        self.assertAlmostEqual(d["score"], 0.9)


# ---------------------------------------------------------------------------
# AudioRelSet unit tests
# ---------------------------------------------------------------------------


class TestAudioRelSet(unittest.TestCase):
    def _make_set(self, records=None):
        return AudioRelSet.from_list(records or SAMPLE_RECORDS)

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
            ds = AudioRelSet.from_manifest(path)
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
                AudioRelSet.from_manifest(path)
        finally:
            os.unlink(path)

    # ---- accessors ----

    def test_relation_types(self):
        ds = self._make_set()
        types = ds.relation_types
        self.assertIn("temporal", types)
        self.assertIn("spatial", types)
        self.assertIn("semantic", types)

    def test_text_prompts(self):
        ds = self._make_set()
        prompts = ds.text_prompts
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all(isinstance(p, str) and len(p) > 0 for p in prompts))

    def test_relations_property_returns_copy(self):
        ds = self._make_set()
        rels_a = ds.relations
        rels_a.append(
            AudioRelation(
                event_a="x", event_b="y", relation_type="temporal", relation="before"
            )
        )
        self.assertEqual(len(ds), 4)

    # ---- __getitem__ (no audio paths) ----

    def test_getitem_no_audio_path(self):
        ds = self._make_set()
        item = ds[0]
        self.assertIn("event_a", item)
        self.assertIn("event_b", item)
        self.assertIn("relation_type", item)
        self.assertIn("relation", item)
        self.assertIn("text_prompt", item)
        self.assertIn("relation_obj", item)
        self.assertNotIn("waveform_a", item)
        self.assertNotIn("waveform_b", item)

    def test_getitem_values(self):
        ds = self._make_set()
        item = ds[0]
        self.assertEqual(item["event_a"], "dog barking")
        self.assertEqual(item["event_b"], "cat meowing")
        self.assertEqual(item["relation_type"], "temporal")
        self.assertEqual(item["relation"], "before")

    # ---- filtering ----

    def test_filter_by_relation_type_temporal(self):
        ds = self._make_set()
        result = ds.filter_by_relation_type("temporal")
        self.assertEqual(len(result), 2)
        for rel in result.relations:
            self.assertEqual(rel.relation_type, "temporal")

    def test_filter_by_relation_type_spatial(self):
        ds = self._make_set()
        result = ds.filter_by_relation_type("spatial")
        self.assertEqual(len(result), 1)

    def test_filter_by_relation_type_invalid_raises(self):
        ds = self._make_set()
        with self.assertRaises(ValueError):
            ds.filter_by_relation_type("random")

    def test_filter_by_relation(self):
        ds = self._make_set()
        result = ds.filter_by_relation("before")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.relations[0].relation, "before")

    def test_filter_by_relation_no_match(self):
        ds = self._make_set()
        result = ds.filter_by_relation("after")
        self.assertEqual(len(result), 0)

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
            self.assertEqual(loaded[0]["event_a"], "dog barking")
        finally:
            os.unlink(path)

    # ---- repr ----

    def test_repr(self):
        ds = self._make_set()
        r = repr(ds)
        self.assertIn("AudioRelSet", r)
        self.assertIn("4", r)

    # ---- default params ----

    def test_default_sample_rate(self):
        ds = AudioRelSet.from_list(SAMPLE_RECORDS)
        self.assertEqual(ds.sample_rate, 22050)

    def test_custom_sample_rate(self):
        ds = AudioRelSet.from_list(SAMPLE_RECORDS, sample_rate=16000)
        self.assertEqual(ds.sample_rate, 16000)

    # ---- vocabulary constants ----

    def test_temporal_relations_covered(self):
        for rel in TEMPORAL_RELATIONS:
            data = {
                "event_a": "a",
                "event_b": "b",
                "relation_type": "temporal",
                "relation": rel,
            }
            obj = AudioRelation.from_dict(data)
            self.assertEqual(obj.relation, rel)

    def test_spatial_relations_covered(self):
        for rel in SPATIAL_RELATIONS:
            data = {
                "event_a": "a",
                "event_b": "b",
                "relation_type": "spatial",
                "relation": rel,
            }
            obj = AudioRelation.from_dict(data)
            self.assertEqual(obj.relation, rel)

    def test_semantic_relations_covered(self):
        for rel in SEMANTIC_RELATIONS:
            data = {
                "event_a": "a",
                "event_b": "b",
                "relation_type": "semantic",
                "relation": rel,
            }
            obj = AudioRelation.from_dict(data)
            self.assertEqual(obj.relation, rel)


if __name__ == "__main__":
    unittest.main()
