# Sandwich Attack Simulation on LMSR-based Blockchain Prediction Markets - Julia Dudlak (jed2206), Trisha Taparia (tdt2128), Yichen Liu (yl6195)

This project simulates sandwich attacks in LMSR-based blockchain prediction markets using trade data from Polymarket. The simulation is used to evaluate attacker profitability, victim harm, and price impact under different market conditions.

The project includes:
- Polymarket trade extraction via Polymarket API
- an LMSR automated market maker implementation
- baseline sandwich attack simulation
- experiments on liquidity parameter b, attacker front-run multiplier, and minimum victim trade size
- visualizations of attack outcomes

The LMSR model is implemented in `lmsr_model.py`, which defines the LMSR cost function, pricing equations, and market transactions.

The Polymarket data extraction is implemented in `polymarket_data.py`, which retrieves trade data from Polymarket APIs (Gamma, CLOB, and Data), specifically for the 2025 NBA Finals, the 2025 World Series, and Super Bowl LX in February 2026. The file also exports cleaned datasets for the simulations.

The sandwich attack simulation framework and experiments are implemented in `sandwich_simulation.py`, where attacker front-running, victim trade execution, and victim back-running are implemented.

[Live Demo Link](https://www.youtube.com/watch?v=6g3hbafEqzI)

---

# Project Structure

```text
project/
│
├── src/
│   ├── polymarket_data.py
│   ├── lmsr_model.py
│   └── sandwich_simulation.py
│
├── polymarket_output/
│   ├── nba-finals-thunder-vs-pacers_trades_clean.csv
│   ├── nba-finals-thunder-vs-pacers_trades_raw.csv
│   ├── nba-finals-thunder-vs-pacers_trades_raw.json
│   ├── nfl-sea-ne-2026-02-08_trades_clean.csv
│   ├── nfl-sea-ne-2026-02-08_trades_raw.csv
│   ├── nfl-sea-ne-2026-02-08_trades_raw.json
│   ├── world-series-winner_trades_clean.csv
│   ├── world-series-winner_trades_raw.csv
│   └── world-series-winner_trades_raw.json
│
├── simulation_output/
|   ├── all_markets_baseline.png
|   ├── all_markets_results.csv
│   ├── NBA_Finals_Thunder_vs_Pacers_baseline.png
|   ├── NBA_Finals_Thunder_vs_Pacers_baseline_results.csv
│   ├── NFL_SEA_vs_NE_baseline.png
|   ├── NFL_SEA_vs_NE_baseline_results.csv
│   ├── World_Series_Winner_baseline.png
|   ├── World_Series_Winner_baseline_results.csv
│   ├── experiment_liquidity.png
|   ├── experiment_liquidity.csv
│   ├── experiment_aggression.png
│   ├── experiment_aggression.csv
│   ├── experiment_victim_threshold.png
│   ├── experiment_victim_threshold.csv
│
└── README.md
```

---

# Installation

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install pandas matplotlib requests
```

---

# Running the Project

## 1. Polymarket Data Extraction

Run:

```bash
python src/polymarket_data.py
```

This creates:

```text
polymarket_output/
```

with cleaned trade datasets for the the 2025 NBA Finals, the 2025 World Series, and Super Bowl LX in February 2026.

---

## 2. Run the Sandwich Attack Simulation + Experiments

Run:

```bash
python src/sandwich_simulation.py
```

This generates CSV summaries, baseline simulation plots, and experiment plots.

Outputs are saved in:

```text
simulation_output/
```

---

