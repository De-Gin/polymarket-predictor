# Sports Predictor

A vector-based probability engine for **NBA game outcomes** and **UFC fight outcomes**, built to find edge against [Polymarket](https://polymarket.com) prices.

---

## Concept

Every piece of data about a matchup becomes a **directional vector** that points toward one outcome or the other. Each vector has four properties:

| Property | Range | Meaning |
|---|---|---|
| `signed_magnitude` | −1 to +1 | How strongly does this data favor outcome B? Negative = favors A |
| `reliability` | 0 to 1 | How trustworthy is this data source historically? |
| `freshness` | 0 to 1 | How recent? Decays exponentially over time |
| `group` | string tag | Which correlation group does this belong to? |

**Effective force** of a vector = `signed_magnitude × reliability × freshness`

All vectors are aggregated in a **VectorSpace**:

1. Vectors sharing a `group` are summed then passed through `tanh(·)` — this prevents 5 correlated signals from counting 5×. A cluster of highly correlated signals contributes at most ±1 force unit, same as a single strong signal.
2. The group-level contributions are summed into a **net force**.
3. Net force is converted to a probability via the **sigmoid function**.

```
net_force = Σ tanh(Σ forces within group)
probability(B wins) = sigmoid(net_force × sensitivity)
```

This directly addresses the independence assumption problem that breaks most naive ensemble models — if your five "different" features are really the same underlying signal, the model won't count them five times.

---

## Sports covered

### NBA — moneyline (who wins)

Vectors used per game:

| Vector | Group | Reliability |
|---|---|---|
| Elo rating differential | `team_strength` | 0.90 |
| L10 recent form differential | `momentum` | 0.55 |
| Home court advantage | `venue` | 0.85 |
| Rest days differential | `schedule` | 0.45 |
| Back-to-back penalty | `schedule` | 0.60 |

Elo is 538-style: K-factor of 20, margin-of-victory multiplier, home-court Elo of 80, 75% season-to-season carryover.

### UFC — fight winner

Vectors used per fight:

| Vector | Group | Reliability |
|---|---|---|
| Elo rating differential | `skill` | 0.85 |
| Last-3 form differential | `skill` | 0.50 |
| Reach differential | `physicality` | 0.55 |
| Age differential (with past-prime penalty) | `physicality` | 0.55 |
| Net striking (landed acc − absorbed) | `striking` | 0.60 |
| Net grappling (TD offense − opponent's TD offense) | `grappling` | 0.55 |
| Layoff / cage rust | `schedule` | 0.40 |

---

## Architecture

```
src/predictor/
├── core/                  # Domain-agnostic vector engine
│   ├── domain.py          # Event, Outcome, EventKind
│   ├── vector.py          # Vector primitive (signed_magnitude × reliability × freshness)
│   ├── vector_space.py    # VectorSpace aggregation → probability + explainability
│   └── math_utils.py      # sigmoid, tanh_saturate, exp_freshness
│
├── nba/                   # NBA adapter
│   ├── elo.py             # 538-style Elo rating system
│   ├── data.py            # nba_api loader + parquet cache
│   ├── features.py        # team state → list[Vector]
│   └── predict.py         # predict_game() orchestration
│
├── ufc/                   # UFC adapter
│   ├── elo.py             # Fighter Elo (no home advantage, title-fight bonus)
│   ├── data.py            # CSV loader
│   ├── features.py        # fighter profile → list[Vector]
│   └── predict.py         # predict_fight() orchestration
│
└── eval/                  # Evaluation tools
    ├── brier.py           # Brier score + skill score
    ├── calibration.py     # Calibration bins + ECE
    ├── backtest.py        # Chronological backtester for NBA + UFC
    └── edge.py            # Polymarket edge = model_prob − market_price, Kelly fraction
```

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Clone the repo
git clone https://github.com/your-username/polymarket-predictot.git
cd polymarket-predictot

# 2. Create virtual environment
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate

# 3. Install
pip install -e ".[dev]"

# 4. Verify
pytest
# Expected: 88 passed
```

---

## Usage

### Predict an NBA game

```python
from datetime import date
from predictor.nba.elo import NbaEloSystem
from predictor.nba.predict import predict_game

# Load season data and train Elo first (see backtest example below)
sys = NbaEloSystem()
# ... record past games ...

pred = predict_game(
    home_team="BOS",
    away_team="LAL",
    game_date=date(2025, 4, 20),
    elo_system=sys,
)

print(pred.summary.readable("Celtics", "Lakers"))
# === PREDICTION ===
#   Celtics: 62.3%
#   Lakers: 37.7%
#   net force: +0.504   agreement: 0.83   vectors: 4
#   top factors:
#     Elo diff (1631 away vs 1589 home)  →  favors Celtics  (force +0.312, group=team_strength)
#     Home court advantage  →  favors Celtics  (force -0.281, group=venue)
```

### Predict a UFC fight

```python
from datetime import date
from predictor.ufc.elo import UfcEloSystem
from predictor.ufc.features import FighterProfile
from predictor.ufc.predict import predict_fight

sys = UfcEloSystem()
# ... record past fights ...

islam = FighterProfile(
    fighter_id="makhachev",
    age=32,
    reach_cm=178,
    slpm=4.2, str_acc=0.57, sapm=2.9, str_def=0.65,
    td_avg=2.8, td_acc=0.45, td_def=0.79,
)
volk = FighterProfile(
    fighter_id="volkanovski",
    age=35,
    reach_cm=182,
    slpm=5.9, str_acc=0.57, sapm=3.6, str_def=0.57,
    td_avg=1.5, td_acc=0.42, td_def=0.66,
)

pred = predict_fight(islam, volk, date(2025, 6, 1), sys,
                     a_name="Makhachev", b_name="Volkanovski")
print(pred.summary.readable("Makhachev", "Volkanovski"))
```

### Compare to Polymarket — find edge

```python
from predictor.eval.edge import edge_vs_market

# Polymarket shows LAL @ BOS: Lakers YES at $0.38 (38% implied probability)
edge = edge_vs_market(
    outcome_name="Lakers win",
    model_probability=0.45,   # your model says 45%
    market_price=0.38,        # Polymarket price
)
print(edge.readable())
# BUY  Lakers win: model=45.0%  market=38.0%  edge=+7.0pp  EV=$+0.043/$  Kelly=11.3%
```

> **Kelly warning:** Always use fractional Kelly (e.g. 0.25× the output) in practice. Full Kelly assumes your probability is exactly correct — it never is.

### Run a full NBA season backtest

```python
from predictor.nba.data import fetch_season_games, merge_to_games
from predictor.eval.backtest import backtest_nba_season
from predictor.eval.brier import brier_score, brier_skill_score
from predictor.eval.calibration import calibration_report

# Fetch from stats.nba.com (cached to data/cache/nba/ after first run)
raw = fetch_season_games("2023-24")
games = merge_to_games(raw)

result = backtest_nba_season(games, skip_warmup_games=200)

probs = result.probabilities
outcomes = result.outcomes

print(f"Games predicted: {len(probs)}")
print(f"Brier score:     {brier_score(probs, outcomes):.4f}  (0.25 = random, lower = better)")
print(f"Skill score:     {brier_skill_score(probs, outcomes):.4f}  (positive = better than base rate)")

cal = calibration_report(probs, outcomes)
print(cal.readable())
```

### Run a UFC backtest

```python
from predictor.ufc.data import load_fights
from predictor.eval.backtest import backtest_ufc
from predictor.eval.brier import brier_score

# Download a UFC fight history CSV (see "Data Sources" below)
fights = load_fights("data/ufc_fights.csv")
result = backtest_ufc(fights, skip_warmup_fights=200)

print(f"Brier: {brier_score(result.probabilities, result.outcomes):.4f}")
```

---

## Data sources

### NBA

Data comes from **[stats.nba.com](https://www.nba.com/stats)** via the `nba_api` Python package. The first call for a season downloads and caches a parquet file in `data/cache/nba/`. Subsequent calls are instant.

```python
from predictor.nba.data import fetch_season_games
games = fetch_season_games("2023-24")  # first call: ~10s. subsequent: instant.
```

### UFC

There's no clean public UFC API. Options:

1. **Kaggle — "Ultimate UFC Dataset"** — search for it on [kaggle.com](https://www.kaggle.com). Download `ufc-master.csv` and rename columns to match the schema in `predictor/ufc/data.py`.
2. **[ufcstats.com](http://www.ufcstats.com/statistics/events/completed)** — the definitive source; scraping requires extra work.

The expected CSV schema is printed by:
```python
from predictor.ufc.data import fights_schema
print(fights_schema())
```

---

## How edge detection works

```
Model probability   →  what the vector engine thinks the true win probability is
Market price        →  Polymarket's current price for the outcome (= implied probability)
Edge                =  model − market

If edge > 0 → the market underprices this outcome → consider buying
If edge < 0 → the market overprices this outcome → consider selling (or ignore)
```

The **Kelly fraction** tells you what percentage of your bankroll the math says to bet, given the edge. Use a fraction of it (25%–33% of Kelly is standard practice) to account for model error.

**This is not financial advice.** The model's calibration must be verified on out-of-sample data (use the backtest + `calibration_report`) before trusting it with real money.

---

## Model evaluation

After a backtest run, check three things:

| Metric | What it tells you | Target |
|---|---|---|
| **Brier score** | Overall accuracy (lower = better, 0.25 = random) | < 0.23 for NBA moneyline |
| **Brier skill score** | How much better than always guessing the base rate | > 0 |
| **ECE (calibration error)** | Does 70% actually happen 70% of the time? | < 0.05 |

If the model is overconfident (predicted 70%, actual 58%), increase the `sensitivity` parameter. If it's underconfident (predicted 55%, actual 68%), increase `sensitivity`. Use a held-out validation season to tune it.

---

## Next steps

- [ ] Tune `sensitivity` per sport via grid search on held-out season
- [ ] Add per-game season NET rating as a vector (requires `nba_api` stats endpoint)
- [ ] Add UFC fighter age trend (career trajectory, not just current age)
- [ ] Add UFC weight-cut difficulty as a vector (catchweight fights)
- [ ] Connect to Polymarket API to pull live market prices automatically
- [ ] Build a daily scheduler to fetch upcoming games → compute edges → output a report
- [ ] Implement Bayesian ELo updating (update mid-fight-card as other results come in)
- [ ] Expand to UFC title fights only (smaller but cleaner calibration dataset)

---

## Tech stack

- **Python 3.11+**
- `nba_api` — NBA data
- `pandas` + `numpy` — data manipulation
- `typer` + `rich` — CLI (coming soon)
- `pytest` — 88 tests, zero external calls
