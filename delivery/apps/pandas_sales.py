"""Analyses a table of sales with pandas.

The analysis a data analyst writes in a notebook on a table of transactions
(time, store, product category, unit price, quantity, discount): clean it,
derive columns, summarise it by group, by day and by store, join a table of
stores, rank and bin. Each step is a common pandas idiom:

- filter out returns and outliers with boolean masks;
- derive revenue, clip discounts, and mark weekends;
- group by category and store with several aggregations;
- a pivot table of revenue by region and category;
- resample to daily revenue, with a seven-day rolling mean;
- merge with a table of stores;
- the ten largest sales, the rank of each store, revenue deciles with
  `pd.qcut` and price bands with `pd.cut`;
- de-duplicate customers and count orders per customer;
- quantiles of basket size.

Every result is reduced to NumPy arrays, so it is compared by its bytes.
"""
from __future__ import annotations

import numpy as np

TITLE = "Analysing a table of sales: filters, group-by, pivot, resample, rolling, merge, rank, binning"
SOURCE = "pandas, called as a library"
LICENCE = None
REQUIRES = ("pandas",)

# By size: rows in the sales table.
ROWS = {"small": 100_000, "large": 2_000_000}
# Stores in the chain.
STORES = 200
# Product categories.
CATEGORIES = 30
# Regions the stores are in.
REGIONS = 8
# Distinct customers, as a fraction of the rows.
CUSTOMER_FRACTION = 0.2
# Days the sales span.
DAYS = 365
# Days in the rolling mean of daily revenue.
ROLLING_DAYS = 7
# Sales whose unit price exceeds this are treated as data-entry errors and dropped.
PRICE_CEILING = 900.0
# Largest discount accepted; larger ones are clipped to it.
MAX_DISCOUNT = 0.5
# Edges of the price bands of `pd.cut`.
PRICE_BANDS = (0.0, 5.0, 20.0, 50.0, 200.0, PRICE_CEILING)
# Number of revenue quantiles of `pd.qcut`.
DECILES = 10
# Sales listed among the largest.
TOP = 10
# Quantiles of basket size reported.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
# Seed of the table.
SEED = 0
# One line per size saying how large its data is.
SIZES = {size: f"{rows:,} sales in {STORES} stores over {DAYS} days" for size, rows in ROWS.items()}


def make_data(size: str) -> tuple:
    """Build a table of sales and a table of stores.

    Args:
        size: One of SIZES.

    Returns:
        (sales as a DataFrame, stores as a DataFrame).
    """
    import pandas as pd

    rows = ROWS[size]
    rng = np.random.default_rng(SEED)
    start = np.datetime64("2025-01-01T00:00:00", "s")
    seconds = np.sort(rng.integers(0, DAYS * 86_400, rows))
    sales = pd.DataFrame({
        "time": start + seconds.astype("timedelta64[s]"),
        "store": rng.integers(0, STORES, rows),
        "category": pd.Categorical.from_codes(rng.integers(0, CATEGORIES, rows),
                                              [f"category {i:02d}" for i in range(CATEGORIES)]),
        "customer": rng.integers(0, int(rows * CUSTOMER_FRACTION), rows),
        "price": np.round(rng.lognormal(2.5, 1.0, rows), 2),
        "quantity": rng.integers(-1, 8, rows),
        "discount": np.round(rng.uniform(0.0, 0.7, rows), 2),
    })
    stores = pd.DataFrame({
        "store": np.arange(STORES),
        "region": [f"region {i}" for i in rng.integers(0, REGIONS, STORES)],
        "floor_area": rng.uniform(200.0, 5_000.0, STORES),
    })
    return sales, stores


def _arrays(frame) -> tuple:
    """Reduce a DataFrame or Series to NumPy arrays: its index, then each column.

    Args:
        frame: A pandas DataFrame or Series.

    Returns:
        A tuple of NumPy arrays; object arrays are compared by their repr.
    """
    if hasattr(frame, "columns"):
        return (frame.index.to_numpy(),) + tuple(frame[column].to_numpy() for column in frame.columns)
    return (frame.index.to_numpy(), frame.to_numpy())


def run(sales, stores) -> tuple:
    """Run the analysis.

    Args:
        sales: The table of sales; not modified.
        stores: The table of stores; not modified.

    Returns:
        Every summary computed, as NumPy arrays.
    """
    import pandas as pd

    # Clean and derive.
    clean = sales[(sales["quantity"] > 0) & (sales["price"] < PRICE_CEILING)].copy()
    clean["discount"] = clean["discount"].clip(upper=MAX_DISCOUNT)
    clean["revenue"] = clean["price"] * clean["quantity"] * (1.0 - clean["discount"])
    clean["weekend"] = clean["time"].dt.dayofweek >= 5

    # Summaries by group.
    by_category = clean.groupby("category", observed=True).agg(
        revenue=("revenue", "sum"), sales=("revenue", "size"), mean_price=("price", "mean"),
        median_quantity=("quantity", "median"))
    by_store = clean.groupby("store").agg(revenue=("revenue", "sum"), customers=("customer", "nunique"),
                                          weekend_share=("weekend", "mean"))

    # Join, pivot.
    joined = clean.merge(stores, on="store", how="left")
    pivot = joined.pivot_table(index="region", columns="category", values="revenue", aggfunc="sum", observed=True)
    by_store = by_store.join(stores.set_index("store"))
    by_store["revenue_per_m2"] = by_store["revenue"] / by_store["floor_area"]
    by_store["rank"] = by_store["revenue_per_m2"].rank(ascending=False)

    # Time series.
    daily = clean.set_index("time")["revenue"].resample("D").sum()
    rolling = daily.rolling(ROLLING_DAYS).mean()

    # Ranking and binning.
    top = clean.nlargest(TOP, "revenue")[["store", "customer", "revenue"]]
    deciles = pd.qcut(clean["revenue"], DECILES, labels=False)
    bands = pd.cut(clean["price"], PRICE_BANDS, labels=False)
    band_revenue = clean.groupby(bands)["revenue"].sum()

    # Customers.
    first_orders = clean.sort_values("time").drop_duplicates("customer")
    orders = clean["customer"].value_counts().sort_index()
    basket = clean.groupby("customer")["quantity"].sum().quantile(list(QUANTILES))

    return (
        _arrays(by_category), _arrays(by_store), _arrays(pivot), _arrays(daily), _arrays(rolling), _arrays(top),
        deciles.to_numpy(), _arrays(band_revenue), _arrays(first_orders[["store", "customer"]]), _arrays(orders),
        _arrays(basket),
    )
