import numpy as np
import pytest


def synthetic_corpus(n_groups=3, per_group=120, dim=64, seed=7):
    """Unit vectors in n_groups well-separated cones, with gold labels and short texts."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_groups, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    V, labels, texts = [], [], []
    for g in range(n_groups):
        pts = centers[g] + 0.15 * rng.normal(size=(per_group, dim))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        V.append(pts)
        labels += [g] * per_group
        texts += [f"Group {g} clause number {i}. This Agreement shall be governed by the laws of the State of Delaware." for i in range(per_group)]
    V = np.vstack(V).astype(np.float32)
    return V, texts, np.array(labels), [f"gold_{g}" for g in range(n_groups)]


@pytest.fixture
def corpus():
    return synthetic_corpus()


@pytest.fixture(autouse=True)
def no_live_keys(monkeypatch):
    """The suite never talks to a model: no key means naming returns its fallback."""
    monkeypatch.delenv("NAMING_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
