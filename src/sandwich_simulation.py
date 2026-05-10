import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from pathlib import Path

from lmsr_model import LMSRMarket


#  Data Loading

def load_trades(csv_path: str, quiet: bool = False) -> pd.DataFrame:
    """
    Load a Polymarket clean CSV, sort chronologically, and filter to
    BUY trades for the dominant outcome (most total volume bought).
    quiet=True suppresses the outcome name print (used during experiments)
    """
    df = pd.read_csv(csv_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    buys = df[df["side"] == "BUY"]
    dominant_index = (
        buys.groupby("outcomeIndex")["size"]
        .sum()
        .idxmax()
    )
    outcome_name = df[df["outcomeIndex"] == dominant_index]["outcome"].iloc[0].strip()

    if not quiet:
        print(f"  Simulating YES = '{outcome_name}' (outcomeIndex {dominant_index})")

    df_yes_buys = df[
        (df["outcomeIndex"] == dominant_index) & (df["side"] == "BUY")
    ].copy().reset_index(drop=True)

    return df_yes_buys


def init_market_from_first_price(df: pd.DataFrame, b: float) -> LMSRMarket:
    """
    Initialize LMSR so p_yes matches the first observed trade price.
    Solves for q_yes using the log-odds transform: q_yes = b * ln(p / (1 - p))
    """
    p = float(df.iloc[0]["price"])
    p = max(0.001, min(0.999, p))
    q_yes = b * math.log(p / (1 - p))
    return LMSRMarket(b=b, q_yes=q_yes, q_no=0.0)


#  Single Sandwich Attack
@dataclass
class AttackResult:
    trade_index:          int
    datetime_utc:         str
    victim_size_usd:      float   # dollars the victim spent
    front_run_size_usd:   float   # dollars the attacker spent front-running
    victim_cost_baseline: float   # what victim would have paid without attack
    victim_cost_actual:   float   # what victim actually paid
    front_run_shares:     float   # YES shares bought by attacker
    back_run_revenue:     float   # dollars attacker received on exit
    attacker_profit:      float   # back_run_revenue - front_run_size_usd
    shares_lost:          float   # shares victim lost due to attack (baseline - actual)
    expected_value_lost:  float   # shares_lost * price_before (dollar impact at current odds)
    price_before:         float
    price_after_frontrun: float
    price_after_victim:   float
    price_after_backrun:  float


def dollars_to_shares(market: LMSRMarket, budget_usd: float) -> float:
    """
    Binary search: how many YES shares can you buy with exactly `budget_usd`?
    """
    lo, hi = 0.0, budget_usd * 10
    for _ in range(60):
        mid = (lo + hi) / 2
        cost = market.cost_to_buy_yes(mid)
        if cost < budget_usd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_single_attack(
    market: LMSRMarket,
    victim_size_usd: float,
    front_run_budget_usd: float,
    trade_index: int,
    datetime_utc,
) -> AttackResult:
    """
    Simulate one sandwich attack at the current market state.
    """
    # Baseline: what would the victim pay with no attacker present?
    baseline_market        = LMSRMarket(b=market.b, q_yes=market.q_yes, q_no=market.q_no)
    victim_shares_baseline = dollars_to_shares(baseline_market, victim_size_usd)
    victim_cost_baseline   = baseline_market.cost_to_buy_yes(victim_shares_baseline)

    price_before = market.price_yes()

    # Step 1: Attacker front-runs (buys YES shares before victim)
    front_run_shares     = dollars_to_shares(market, front_run_budget_usd)
    front_run_cost       = market.buy_yes(front_run_shares)
    price_after_frontrun = market.price_yes()

    # Step 2: Victim buys YES shares at the now-inflated price
    victim_shares        = dollars_to_shares(market, victim_size_usd)
    victim_cost_actual   = market.buy_yes(victim_shares)
    price_after_victim   = market.price_yes()

    # Step 3: Attacker back-runs (sells front-run shares at higher price)
    shares_to_sell      = max(0.0, min(front_run_shares, market.q_yes))
    back_run_revenue    = market.sell_yes(shares_to_sell) if shares_to_sell > 0 else 0.0
    price_after_backrun = market.price_yes()

    attacker_profit     = back_run_revenue - front_run_cost
    shares_lost         = victim_shares_baseline - victim_shares
    expected_value_lost = shares_lost * price_before

    return AttackResult(
        trade_index          = trade_index,
        datetime_utc         = str(datetime_utc),
        victim_size_usd      = victim_size_usd,
        front_run_size_usd   = front_run_cost,
        victim_cost_baseline = victim_cost_baseline,
        victim_cost_actual   = victim_cost_actual,
        front_run_shares     = front_run_shares,
        back_run_revenue     = back_run_revenue,
        attacker_profit      = attacker_profit,
        shares_lost          = shares_lost,
        expected_value_lost  = expected_value_lost,
        price_before         = price_before,
        price_after_frontrun = price_after_frontrun,
        price_after_victim   = price_after_victim,
        price_after_backrun  = price_after_backrun,
    )


#  Full Simulation Over Trade History
def run_simulation(
    csv_path: str,
    b: float = 500.0,
    front_run_multiplier: float = 1.0,
    min_victim_size_usd: float = 50.0,
    quiet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run sandwich attack simulation over all qualifying trades in a CSV.
    Args:
        csv_path:             path to Polymarket clean CSV
        b:                    LMSR liquidity parameter
        front_run_multiplier: attacker spends this multiple of victim size
        min_victim_size_usd:  only attack trades at or above this threshold
        quiet:                suppress outcome name print (used during experiments)
    Returns:
        results_df:  one row per attack with all metrics
        trades_df:   filtered trade history used in simulation
    """
    trades_df = load_trades(csv_path, quiet=quiet)
    market    = init_market_from_first_price(trades_df, b)

    results = []

    for i, row in trades_df.iterrows():
        victim_size = float(row["size"])

        if victim_size >= min_victim_size_usd:
            front_run_budget = victim_size * front_run_multiplier
            result = simulate_single_attack(
                market = market,
                victim_size_usd = victim_size,
                front_run_budget_usd = front_run_budget,
                trade_index = i,
                datetime_utc = row["datetime_utc"],
            )
            results.append(result)
        else:
            # Small trade: no attack (base min $50 for now)
            shares = dollars_to_shares(market, victim_size)
            market.buy_yes(shares)

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df["datetime_utc"] = pd.to_datetime(results_df["datetime_utc"], utc=True)

    return results_df, trades_df


#  Baseline Output
def print_summary(results_df: pd.DataFrame, market_name: str):
    print(f"\n{'='*55}")
    print(f"  {market_name}")
    print(f"{'='*55}")
    print(f"  Attacks simulated:         {len(results_df)}")
    print(f"  Profitable attacks:        {(results_df['attacker_profit'] > 0).sum()}")
    print(f"  Total attacker profit:     ${results_df['attacker_profit'].sum():.4f}")
    print(f"  Avg profit per attack:     ${results_df['attacker_profit'].mean():.4f}")
    print(f"  Max profit (single):       ${results_df['attacker_profit'].max():.4f}")
    print(f"  Total shares lost:         {results_df['shares_lost'].sum():.4f} shares")
    print(f"  Avg shares lost:           {results_df['shares_lost'].mean():.4f} shares")
    print(f"  Total expected value lost: ${results_df['expected_value_lost'].sum():.4f}")
    print(f"  Avg expected value lost:   ${results_df['expected_value_lost'].mean():.4f}")
    print(f"  Avg price impact:          {(results_df['price_after_frontrun'] - results_df['price_before']).mean():.4f}")

def plot_baseline(results_df: pd.DataFrame, market_name: str, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Sandwich Attack Simulation — {market_name}", fontsize=14)

    # 1. Attacker profit over time
    ax = axes[0, 0]
    ax.scatter(results_df["datetime_utc"], results_df["attacker_profit"],
               s=20, alpha=0.6, color="steelblue")
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.set_title("Attacker Profit per Attack")
    ax.set_xlabel("Date")
    ax.set_ylabel("Profit (USDC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # 2. Shares lost by victim vs victim trade size
    ax = axes[0, 1]
    ax.scatter(results_df["victim_size_usd"], results_df["shares_lost"],
               s=20, alpha=0.6, color="firebrick")
    ax.set_title("Victim Shares Lost vs Trade Size")
    ax.set_xlabel("Victim Trade Size (USDC)")
    ax.set_ylabel("Shares Lost Due to Attack")
    ax.axhline(0, color="gray", lw=1, linestyle="--")

    # 3. Cumulative attacker profit
    ax = axes[1, 0]
    cumulative = results_df["attacker_profit"].cumsum()
    ax.plot(results_df["datetime_utc"], cumulative, color="steelblue", lw=2)
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.set_title("Cumulative Attacker Profit")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit (USDC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # 4. Price impact of front-run vs front-run size
    ax = axes[1, 1]
    price_impact = results_df["price_after_frontrun"] - results_df["price_before"]
    ax.scatter(results_df["front_run_size_usd"], price_impact,
               s=20, alpha=0.6, color="darkorange")
    ax.set_title("Price Impact of Front-Run")
    ax.set_xlabel("Front-Run Size (USDC)")
    ax.set_ylabel("Price Increase (YES)")

    plt.tight_layout()
    out_path = output_dir / f"{market_name.replace(' ', '_')}_baseline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


#  Experiment 1: Liquidity Parameter b

def experiment_liquidity(markets: list, output_dir: Path) -> pd.DataFrame:
    """
    Vary b across a range while holding front_run_multiplier=1.0 and min_victim_size_usd=50.0
    """
    b_values = [50, 100, 250, 500, 1000, 2000]
    rows = []

    for m in markets:
        for b in b_values:
            results_df, _ = run_simulation(m["csv"], b=b, quiet=True)
            rows.append({
                "market":               m["name"],
                "b":                    b,
                "avg_attacker_profit":  results_df["attacker_profit"].mean(),
                "total_attacker_profit":results_df["attacker_profit"].sum(),
                "profitable_rate":      (results_df["attacker_profit"] > 0).mean(),
                "avg_ev_lost":          results_df["expected_value_lost"].mean(),
                "total_ev_lost":        results_df["expected_value_lost"].sum(),
                "avg_price_impact":     (results_df["price_after_frontrun"] - results_df["price_before"]).mean(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "experiment_liquidity.csv", index=False)

    # Plot: b vs avg attacker profit and b vs avg price impact, one line per market
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 1: Liquidity Parameter vs Attack Outcomes", fontsize=13)
    colors = {"NBA Finals Thunder vs Pacers": "steelblue",
              "NFL SEA vs NE": "firebrick",
              "World Series Winner": "darkorange"}

    for market_name, group in df.groupby("market"):
        axes[0].plot(group["b"], group["avg_attacker_profit"],
                     marker="o", lw=2, label=market_name, color=colors.get(market_name))
        axes[1].plot(group["b"], group["avg_price_impact"],
                     marker="o", lw=2, label=market_name, color=colors.get(market_name))

    axes[0].axhline(0, color="gray", lw=1, linestyle="--")
    axes[0].set_title("Liquidity Parameter vs Avg Attacker Profit")
    axes[0].set_xlabel("Liquidity Parameter b")
    axes[0].set_ylabel("Avg Attacker Profit (USDC)")
    axes[0].legend()

    axes[1].set_title("Liquidity Parameter vs Avg Price Impact")
    axes[1].set_xlabel("Liquidity Parameter b")
    axes[1].set_ylabel("Avg Price Impact")
    axes[1].legend()

    plt.tight_layout()
    out_path = output_dir / "experiment_liquidity.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    return df


#  Experiment 2: Attacker Aggressiveness
def experiment_aggression(markets: list, output_dir: Path) -> pd.DataFrame:
    """
    Vary front_run_multiplier while holding b=500 and min_victim_size_usd=50.0
    """
    multipliers = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []

    for m in markets:
        for mult in multipliers:
            results_df, _ = run_simulation(m["csv"], front_run_multiplier=mult, quiet=True)
            rows.append({
                "market":                m["name"],
                "front_run_multiplier":  mult,
                "avg_attacker_profit":   results_df["attacker_profit"].mean(),
                "total_attacker_profit": results_df["attacker_profit"].sum(),
                "profitable_rate":       (results_df["attacker_profit"] > 0).mean(),
                "avg_ev_lost":           results_df["expected_value_lost"].mean(),
                "total_ev_lost":         results_df["expected_value_lost"].sum(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "experiment_aggression.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 2: Attacker Aggressiveness vs Attack Outcomes", fontsize=13)
    colors = {"NBA Finals Thunder vs Pacers": "steelblue",
              "NFL SEA vs NE": "firebrick",
              "World Series Winner": "darkorange"}

    for market_name, group in df.groupby("market"):
        axes[0].plot(group["front_run_multiplier"], group["avg_attacker_profit"],
                     marker="o", lw=2, label=market_name, color=colors.get(market_name))
        axes[1].plot(group["front_run_multiplier"], group["profitable_rate"],
                     marker="o", lw=2, label=market_name, color=colors.get(market_name))

    axes[0].axhline(0, color="gray", lw=1, linestyle="--")
    axes[0].set_title("Front-Run Multiplier vs Avg Attacker Profit")
    axes[0].set_xlabel("Front-Run Multiplier")
    axes[0].set_ylabel("Avg Attacker Profit (USDC)")
    axes[0].legend()

    axes[1].set_title("Front-Run Multiplier vs Profitable Attack Rate")
    axes[1].set_xlabel("Front-Run Multiplier")
    axes[1].set_ylabel("Proportion of Profitable Attacks")
    axes[1].legend()

    plt.tight_layout()
    out_path = output_dir / "experiment_aggression.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    return df


#  Experiment 3: Minimum Victim Trade Size
def experiment_victim_threshold(markets: list, output_dir: Path) -> pd.DataFrame:
    """
    Vary min_victim_size_usd while holding b=500 and front_run_multiplier=1.0
    """
    thresholds = [0, 10, 25, 50, 100, 250, 500, 1000]
    rows = []

    for m in markets:
        for threshold in thresholds:
            results_df, _ = run_simulation(
                m["csv"], min_victim_size_usd=threshold, quiet=True
            )
            rows.append({
                "market":                m["name"],
                "min_victim_size_usd":   threshold,
                "attacks_simulated":     len(results_df),
                "total_attacker_profit": results_df["attacker_profit"].sum(),
                "avg_attacker_profit":   results_df["attacker_profit"].mean(),
                "profitable_rate":       (results_df["attacker_profit"] > 0).mean(),
                "total_ev_lost":         results_df["expected_value_lost"].sum(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "experiment_victim_threshold.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 3: Minimum Victim Trade Size vs Attack Outcomes", fontsize=13)
    colors = {"NBA Finals Thunder vs Pacers": "steelblue",
              "NFL SEA vs NE": "firebrick",
              "World Series Winner": "darkorange"}

    for market_name, group in df.groupby("market"):
        axes[0].plot(group["min_victim_size_usd"], group["total_attacker_profit"],
                     marker="o", lw=2, label=market_name, color=colors.get(market_name))
        axes[1].plot(group["min_victim_size_usd"], group["profitable_rate"],
                     marker="o", lw=2, label=market_name, color=colors.get(market_name))

    axes[0].set_title("Victim Threshold vs Total Attacker Profit")
    axes[0].set_xlabel("Minimum Victim Trade Size (USDC)")
    axes[0].set_ylabel("Total Attacker Profit (USDC)")
    axes[0].legend()

    axes[1].set_title("Victim Threshold vs Profitable Attack Rate")
    axes[1].set_xlabel("Minimum Victim Trade Size (USDC)")
    axes[1].set_ylabel("Proportion of Profitable Attacks")
    axes[1].legend()

    plt.tight_layout()
    out_path = output_dir / "experiment_victim_threshold.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    return df


if __name__ == "__main__":

    markets = [
        {
            "name": "NBA Finals Thunder vs Pacers",
            "csv":  "polymarket_output/nba-finals-thunder-vs-pacers_trades_clean.csv",
        },
        {
            "name": "NFL SEA vs NE",
            "csv":  "polymarket_output/nfl-sea-ne-2026-02-08_trades_clean.csv",
        },
        {
            "name": "World Series Winner",
            "csv":  "polymarket_output/world-series-winner_trades_clean.csv",
        },
    ]

    B = 500.0
    FRONT_RUN_MULT = 1.0
    MIN_VICTIM_SIZE_USD = 50.0

    output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)

    # Baseline run
    print("=" * 55)
    print("  BASELINE SIMULATION")
    print("=" * 55)

    all_results = []

    for m in markets:
        print(f"\nRunning baseline: {m['name']}")
        results_df, trades_df = run_simulation(
            csv_path = m["csv"],
            b = B,
            front_run_multiplier = FRONT_RUN_MULT,
            min_victim_size_usd  = MIN_VICTIM_SIZE_USD,
        )
        print_summary(results_df, m["name"])
        plot_baseline(results_df, m["name"], output_dir)

        results_df["market"] = m["name"]
        all_results.append(results_df)

        out_csv = output_dir / f"{m['name'].replace(' ', '_')}_baseline_results.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"  Results saved: {out_csv}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "all_markets_baseline.csv", index=False)
    print("\nBaseline combined results saved.")

    # Experiments
    print("\n" + "=" * 55)
    print("  EXPERIMENT 1: LIQUIDITY PARAMETER b")
    print("=" * 55)
    liq_df = experiment_liquidity(markets, output_dir)

    print("\n" + "=" * 55)
    print("  EXPERIMENT 2: ATTACKER AGGRESSIVENESS")
    print("=" * 55)
    agg_df = experiment_aggression(markets, output_dir)

    print("\n" + "=" * 55)
    print("  EXPERIMENT 3: MINIMUM VICTIM TRADE SIZE")
    print("=" * 55)
    vic_df = experiment_victim_threshold(markets, output_dir)

    print("\nExperiments complete.")