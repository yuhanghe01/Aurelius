"""
Tests for the Aurelius model.
"""

import unittest

import torch

from aurelius.models.aurelius import (
    Aurelius,
    ConditioningProjector,
    RelationEncoder,
    UNet1D,
    _sinusoidal_embedding,
)


class TestSinusoidalEmbedding(unittest.TestCase):
    def test_output_shape(self):
        t = torch.arange(4)
        emb = _sinusoidal_embedding(t, 64)
        self.assertEqual(emb.shape, (4, 64))

    def test_values_finite(self):
        t = torch.tensor([0, 100, 500, 999])
        emb = _sinusoidal_embedding(t, 128)
        self.assertTrue(torch.isfinite(emb).all())


class TestRelationEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = RelationEncoder(embed_dim=64)
        types = ["temporal", "spatial", "semantic"]
        rels = ["before", "left", "similar"]
        out = enc(types, rels)
        self.assertEqual(out.shape, (3, 64))

    def test_different_inputs_differ(self):
        enc = RelationEncoder(embed_dim=64)
        out1 = enc(["temporal"], ["before"])
        out2 = enc(["temporal"], ["after"])
        self.assertFalse(torch.allclose(out1, out2))


class TestConditioningProjector(unittest.TestCase):
    def test_output_shape(self):
        proj = ConditioningProjector(text_dim=32, rel_dim=16, out_dim=64)
        text_emb = torch.randn(4, 32)
        rel_emb = torch.randn(4, 16)
        out = proj(text_emb, rel_emb)
        self.assertEqual(out.shape, (4, 64))


class TestUNet1D(unittest.TestCase):
    def _make_unet(self):
        return UNet1D(
            in_channels=4,
            base_channels=16,
            num_levels=2,
            cond_dim=32,
            time_embed_dim=32,
        )

    def test_output_shape(self):
        unet = self._make_unet()
        B, C, T = 2, 4, 32
        x = torch.randn(B, C, T)
        t = torch.randint(0, 100, (B,))
        cond = torch.randn(B, 32)
        out = unet(x, t, cond)
        self.assertEqual(out.shape, (B, C, T))

    def test_output_finite(self):
        unet = self._make_unet()
        x = torch.randn(2, 4, 32)
        t = torch.zeros(2, dtype=torch.long)
        cond = torch.randn(2, 32)
        out = unet(x, t, cond)
        self.assertTrue(torch.isfinite(out).all())


class TestAurelius(unittest.TestCase):
    def _make_model(self):
        return Aurelius(
            text_dim=32,
            rel_embed_dim=16,
            cond_dim=32,
            latent_channels=4,
            latent_length=16,
            num_diffusion_steps=10,
            unet_base_channels=16,
            unet_num_levels=2,
        )

    def test_encode_text(self):
        model = self._make_model()
        texts = ["dog barking before cat meowing", "thunder and rain simultaneously"]
        emb = model.encode_text(texts)
        self.assertEqual(emb.shape, (2, 32))

    def test_forward_returns_cond_emb(self):
        model = self._make_model()
        text_emb = torch.randn(2, 32)
        types = ["temporal", "spatial"]
        rels = ["before", "left"]
        output = model(text_emb, types, rels)
        self.assertIn("cond_emb", output)
        self.assertEqual(output["cond_emb"].shape, (2, 32))
        self.assertNotIn("loss", output)

    def test_forward_with_targets_returns_loss(self):
        model = self._make_model()
        text_emb = torch.randn(2, 32)
        types = ["temporal", "semantic"]
        rels = ["after", "similar"]
        targets = torch.randn(2, 4, 16)
        output = model(text_emb, types, rels, target_latents=targets)
        self.assertIn("loss", output)
        self.assertIn("noise_pred", output)
        loss = output["loss"]
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))

    def test_generate_output_shape(self):
        model = self._make_model()
        text_emb = torch.randn(2, 32)
        types = ["temporal", "spatial"]
        rels = ["simultaneous", "right"]
        latents = model.generate(text_emb, types, rels, num_steps=3)
        self.assertEqual(latents.shape, (2, 4, 16))

    def test_generate_is_deterministic_with_seed(self):
        model = self._make_model()
        text_emb = torch.randn(1, 32)
        types = ["semantic"]
        rels = ["contrasting"]

        torch.manual_seed(0)
        out1 = model.generate(text_emb, types, rels, num_steps=2)
        torch.manual_seed(0)
        out2 = model.generate(text_emb, types, rels, num_steps=2)
        self.assertTrue(torch.allclose(out1, out2))

    def test_loss_backward(self):
        model = self._make_model()
        text_emb = torch.randn(2, 32)
        types = ["temporal", "semantic"]
        rels = ["before", "complementary"]
        targets = torch.randn(2, 4, 16)
        output = model(text_emb, types, rels, target_latents=targets)
        output["loss"].backward()
        # Check that at least one gradient was computed
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())
                return
        self.fail("No gradients were computed")


if __name__ == "__main__":
    unittest.main()
