# -*- coding: utf-8 -*-
"""
generate_synthetic_data.py

Generates a synthetic influencer dataset with GENUINE, mechanistically-grounded
fraud patterns baked into the labels (as opposed to the random `is_fraud`
fallback in the original submission, which produced labels with zero real
relationship to the features).

This script only exists because the original raw `influencers.csv` was not
included in the delivered archive -- only the model's *output* CSV was.
To make the corrected pipeline honestly demonstrable and testable end-to-end,
this generator stands in for that missing raw input. If a real labeled
dataset exists, point fraud_detection.py at that file instead and skip this
script entirely.

Three influencer archetypes are simulated:
  1. Organic (legit)      -- ~88% of accounts. Engagement, comments and
                              growth all scale naturally with followers.
  2. Bought-follower fraud -- follower count inflated via purchase; likes/
                              comments do NOT scale with followers, so
                              engagement_rate collapses.
  3. Bot-engagement fraud  -- engagement is inflated via comment/like bots;
                              comment/like ratio is abnormally high, and
                              growth_rate shows unnatural spikes uncorrelated
                              with account age.

Because fraud is defined by an actual generative mechanism (not a coin
flip), a model that reads the engineered ratio features has real signal to
find -- which is what "validation" is supposed to measure.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000
FRAUD_RATE = 0.12  # ~12% of accounts are fraudulent, split across two archetypes


def _organic_batch(n):
    followers = RNG.lognormal(mean=13.0, sigma=1.1, size=n).clip(1000, 5_000_000)
    account_age_days = RNG.integers(60, 2000, size=n)
    # Real engagement rate: healthy accounts sit ~1-6%, decaying slightly for mega-accounts
    base_engagement = RNG.normal(3.2, 1.1, size=n).clip(0.4, 8.0)
    base_engagement *= (1.0 - 0.15 * np.log10(followers / 1000).clip(0, 4) / 4)
    likes = (followers * (base_engagement / 100) * RNG.normal(1.0, 0.15, size=n)).clip(10)
    comments = (likes * RNG.normal(0.045, 0.015, size=n)).clip(1)
    growth_rate = RNG.normal(3.0, 4.0, size=n).clip(-15, 25)
    engagement_rate = (likes / followers) * 100
    return followers, likes, comments, growth_rate, engagement_rate, account_age_days


def _bought_followers_batch(n):
    # Followers inflated via purchase -> high follower count, young-ish account,
    # but likes/comments stay at "organic-for-a-much-smaller-account" levels.
    followers = RNG.lognormal(mean=13.6, sigma=1.0, size=n).clip(5000, 5_000_000)
    account_age_days = RNG.integers(30, 900, size=n)
    true_organic_base = RNG.lognormal(mean=9.5, sigma=0.8, size=n)  # a much smaller "real" audience
    likes = (true_organic_base * RNG.normal(0.03, 0.01, size=n).clip(0.005)).clip(5)
    comments = (likes * RNG.normal(0.03, 0.01, size=n)).clip(0)
    growth_rate = RNG.normal(18.0, 12.0, size=n).clip(-5, 60)  # sudden follower spikes from purchase
    engagement_rate = (likes / followers) * 100  # collapses because followers are fake
    return followers, likes, comments, growth_rate, engagement_rate, account_age_days


def _bot_engagement_batch(n):
    # Engagement inflated via bots -> comment/like ratio is abnormally high
    # and disproportionate to a believable human audience.
    followers = RNG.lognormal(mean=12.6, sigma=1.0, size=n).clip(2000, 3_000_000)
    account_age_days = RNG.integers(20, 1200, size=n)
    engagement_rate = RNG.normal(3.0, 1.0, size=n).clip(0.5, 7.0)
    likes = (followers * (engagement_rate / 100)).clip(10)
    comments = (likes * RNG.normal(0.35, 0.12, size=n).clip(0.15))  # bots spam comments disproportionately
    growth_rate = RNG.normal(9.0, 10.0, size=n).clip(-10, 45)
    return followers, likes, comments, growth_rate, engagement_rate, account_age_days


def generate(n=N, fraud_rate=FRAUD_RATE, seed=42):
    global RNG
    RNG = np.random.default_rng(seed)

    n_fraud = int(round(n * fraud_rate))
    n_bought = n_fraud // 2
    n_bot = n_fraud - n_bought
    n_organic = n - n_fraud

    rows = []
    for batch_fn, count, label in [
        (_organic_batch, n_organic, 0),
        (_bought_followers_batch, n_bought, 1),
        (_bot_engagement_batch, n_bot, 1),
    ]:
        f, l, c, g, e, a = batch_fn(count)
        for i in range(count):
            rows.append((f[i], l[i], c[i], g[i], e[i], a[i], label))

    df = pd.DataFrame(rows, columns=[
        "followers", "likes", "comments", "growth_rate",
        "engagement_rate", "account_age_days", "is_fraud"
    ])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "name", [f"influencer_{i+1}" for i in range(len(df))])
    df["followers"] = df["followers"].round().astype(int)
    df["likes"] = df["likes"].round().astype(int)
    df["comments"] = df["comments"].round().astype(int)
    df["growth_rate"] = df["growth_rate"].round(2)
    df["engagement_rate"] = df["engagement_rate"].round(2)
    df["account_age_days"] = df["account_age_days"].astype(int)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/v2/data/influencers.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["is_fraud"].value_counts(normalize=True))
