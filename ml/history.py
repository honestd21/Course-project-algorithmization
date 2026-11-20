import numpy as np
from simulator.field import Field


def neighbor_mean(weed, obstacles):
    """Среднее засорение по 4 соседям (только существующие клетки, препятствия считаем нулём)."""
    w = np.where(obstacles, 0.0, weed)
    s = np.zeros_like(w)
    cnt = np.zeros_like(w)
    s[1:, :] += w[:-1, :];  cnt[1:, :] += 1
    s[:-1, :] += w[1:, :];  cnt[:-1, :] += 1
    s[:, 1:] += w[:, :-1];  cnt[:, 1:] += 1
    s[:, :-1] += w[:, 1:];  cnt[:, :-1] += 1
    return s / np.maximum(cnt, 1)


def step_weed(weed, obstacles, rng, spread=0.25, growth=0.08, noise=0.02):
    """Один шаг динамики: сорняки расползаются к соседям, растут и чуть шумят."""
    nb = neighbor_mean(weed, obstacles)
    inflow = spread * np.maximum(nb - weed, 0)
    new = weed + inflow + growth * weed * (1 - weed)
    new += rng.normal(0, noise, weed.shape)
    new = np.clip(new, 0, 1)
    new[obstacles] = 0
    return new


def generate_history(n_fields=50, steps=10, H=30, W=30, seed=0):
    """Возвращает список (field_id, [weed_0, ..., weed_steps], obstacles)."""
    rng = np.random.default_rng(seed)
    history = []
    for fid in range(n_fields):
        f = Field.random(H, W, seed=int(rng.integers(1_000_000_000)))
        frames = [f.weed.copy()]
        for _ in range(steps):
            frames.append(step_weed(frames[-1], f.obstacles, rng))
        history.append((fid, frames, f.obstacles))
    return history