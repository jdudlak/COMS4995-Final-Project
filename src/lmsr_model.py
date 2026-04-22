import math

class LMSRMarket:
    """
    b: liquidity parameter (higher b = more liquid)
    q_yes: outstanding YES shares (default to 0)
    q_no: outstanding NO shares (default to 0)
    """

    def __init__(self, b: float, q_yes: float = 0.0, q_no: float = 0.0):
        if b <= 0:
            raise ValueError("b must be positive")
        self.b = b
        self.q_yes = q_yes
        self.q_no = q_no

    def cost_function(self, q_yes: float, q_no: float) -> float:
        """
        LMSR cost function: C(q_yes, q_no) = b · ln(e^(q_yes/b) + e^(q_no/b))
        """
        a = q_yes / self.b
        c = q_no  / self.b
        m = max(a, c)  # so we don't overflow float when q_yes/b or q_no/b are large
        return self.b * (m + math.log(math.exp(a - m) + math.exp(c - m)))

    def price_yes(self, q_yes: float | None = None,
                        q_no:  float | None = None) -> float:
        """
        Instantaneous price of a YES share: exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))

        made it with optional q_yes / q_no so we can check the price at a
        hypothetical state if needed
        """
        qy = q_yes if q_yes is not None else self.q_yes
        qn = q_no  if q_no  is not None else self.q_no
        a  = qy / self.b
        c  = qn / self.b
        m  = max(a, c)
        exp_a = math.exp(a - m)
        exp_c = math.exp(c - m)
        return exp_a / (exp_a + exp_c)

    def price_no(self, q_yes: float | None = None,
                       q_no:  float | None = None) -> float:
        """Instantaneous price of a NO share: 1 - p_yes"""
        return 1.0 - self.price_yes(q_yes, q_no)

    def cost_to_buy_yes(self, amt: float) -> float:
        """
        Cost to purchase given amount of YES shares at the current state
        = C(q_yes + delta, q_no) - C(q_yes, q_no)
        """
        if amt < 0:
            raise ValueError("Enter positive number of shares")
        return (self.cost_function(self.q_yes + amt, self.q_no)
                - self.cost_function(self.q_yes, self.q_no))

    def cost_to_sell_yes(self, amt: float) -> float:
        """
        Revenue from selling given amount of YES shares at the current state
        = C(q_yes, q_no) - C(q_yes - delta, q_no)
        """
        if amt < 0:
            raise ValueError("Enter positive number of shares")
        if amt > self.q_yes:
            raise ValueError("Can't sell more YES shares than currently held")
        return (self.cost_function(self.q_yes, self.q_no)
                - self.cost_function(self.q_yes - amt, self.q_no))

    def cost_to_buy_no(self, amt: float) -> float:
        """Cost to purchase given amount of NO shares"""
        if amt < 0:
            raise ValueError("Use cost_to_sell_no for sales.")
        return (self.cost_function(self.q_yes, self.q_no + amt)
                - self.cost_function(self.q_yes, self.q_no))

    def cost_to_sell_no(self, amt: float) -> float:
        """Revenue from selling given amount of NO shares"""
        if amt < 0:
            raise ValueError("Enter positive number of shares")
        if amt > self.q_no:
            raise ValueError("Can't sell more NO shares than currently held")
        return (self.cost_function(self.q_yes, self.q_no)
                - self.cost_function(self.q_yes, self.q_no - amt))

    def buy_yes(self, amt: float) -> float:
        cost = self.cost_to_buy_yes(amt)
        self.q_yes += amt
        return cost

    def sell_yes(self, amt: float) -> float:
        revenue = self.cost_to_sell_yes(amt)
        self.q_yes -= amt
        return revenue

    def buy_no(self, amt: float) -> float:
        cost = self.cost_to_buy_no(amt)
        self.q_no += amt
        return cost

    def sell_no(self, amt: float) -> float:
        revenue = self.cost_to_sell_no(amt)
        self.q_no -= amt
        return revenue

    def state(self) -> dict:
        """to get current market state info"""
        return {
            "b":         self.b,
            "q_yes":     self.q_yes,
            "q_no":      self.q_no,
            "price_yes": self.price_yes(),
            "price_no":  self.price_no(),
        }

    def __repr__(self) -> str:
        """formats current market state nicely lol"""
        s = self.state()
        return (f"LMSRMarket(b={s['b']}, q_yes={s['q_yes']:.4f}, "
                f"q_no={s['q_no']:.4f}, "
                f"p_yes={s['price_yes']:.4f})")


if __name__ == "__main__":

    market = LMSRMarket(b=100)
    print(f"\nInitial state:  {market}")

    # At q_yes = q_no = 0, price should be exactly 0.5
    assert abs(market.price_yes() - 0.5) < 1e-10, "Initial price should be 0.5"
    print("✓  Initial p_yes = 0.5  (correct for balanced market)")

    # Buy 50 YES shares
    cost = market.buy_yes(50)
    print(f"\nBought 50 YES shares — cost: ${cost:.4f}")
    print(f"State after:    {market}")
    assert market.price_yes() > 0.5, "Price should rise after buying YES"
    print("✓  p_yes rose above 0.5 after buying YES")

    # Prices must still sum to 1
    assert abs(market.price_yes() + market.price_no() - 1.0) < 1e-10
    print("✓  p_yes + p_no = 1.0")

    # Sell those shares back — should recover (nearly) all the money
    revenue = market.sell_yes(50)
    print(f"\nSold 50 YES shares back — revenue: ${revenue:.4f}")
    print(f"State after:    {market}")
    assert abs(market.price_yes() - 0.5) < 1e-10, "Price should return to 0.5"
    assert abs(cost - revenue) < 1e-10, "Buy cost should equal sell revenue for same delta"
    print("✓  Price returned to 0.5")
    print("✓  Buy cost == sell revenue (reversible trades)")

    # Larger b → smaller price impact for the same trade
    print("\n--- Effect of liquidity parameter b ---")
    for b in [50, 100, 200]:
        m = LMSRMarket(b=b)
        c = m.cost_to_buy_yes(50)
        p_after = m.price_yes(q_yes=50)
        print(f"  b={b:>4}  cost of 50 YES: ${c:>7.4f}   "
              f"price after trade: {p_after:.4f}")

    print("\nAll checks passed.")
