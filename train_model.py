from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Treat impossible negative weights as missing.
    if "weight" in x:
        x.loc[x["weight"] <= 0, "weight"] = np.nan

    d = pd.to_datetime(x["date"])
    x["year"] = d.dt.year
    x["month"] = d.dt.month
    x["day"] = d.dt.day
    x["dow"] = d.dt.dayofweek
    x["doy"] = d.dt.dayofyear
    x["weekofyear"] = d.dt.isocalendar().week.astype(int)
    x["month_sin"] = np.sin(2 * np.pi * x["doy"] / 365.25)
    x["month_cos"] = np.cos(2 * np.pi * x["doy"] / 365.25)
    x["day_index"] = (d - pd.Timestamp("2025-01-01")).dt.days

    # Route geometry. December input does not contain coordinates; they are
    # added before this function from the city-coordinate lookup.
    lat1 = np.radians(x["pickup_lat"])
    lat2 = np.radians(x["delivery_lat"])
    lon1 = np.radians(x["pickup_lon"])
    lon2 = np.radians(x["delivery_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    x["geo_distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    x["lat_diff"] = x["delivery_lat"] - x["pickup_lat"]
    x["lon_diff"] = x["delivery_lon"] - x["pickup_lon"]
    x["abs_lat_diff"] = x["lat_diff"].abs()
    x["abs_lon_diff"] = x["lon_diff"].abs()
    x["distance_per_geo"] = x["distance"] / (x["geo_distance"] + 1)

    return x.drop(columns=["load_id", "date", "posted_rate"], errors="ignore")


def add_missing_december_columns(dec: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()

    # Build city -> coordinates from labeled development + validation data.
    pickup = reference[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery = reference[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    coords = pd.concat([pickup, delivery], ignore_index=True).drop_duplicates("city")
    lookup = coords.set_index("city")[["lat", "lon"]].to_dict("index")

    for side in ["pickup", "delivery"]:
        out[f"{side}_lat"] = out[side].map(lambda c: lookup[c]["lat"])
        out[f"{side}_lon"] = out[side].map(lambda c: lookup[c]["lon"])

    # These fields are absent from the supplied December file; keep them as
    # missing rather than inventing values.
    out["market_index"] = np.nan
    out["quote_signal"] = np.nan
    out["load_id"] = "DEC-" + out.index.astype(str)
    out["posted_rate"] = np.nan
    return out


def fit_and_predict(train: pd.DataFrame, target: pd.Series, x_pred: pd.DataFrame,
                    cat_indices: list[int], seeds=(17, 42, 73),
                    iterations=260) -> np.ndarray:
    predictions = []
    for seed in seeds:
        model = CatBoostRegressor(
            iterations=iterations,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            l2_leaf_reg=8,
            random_strength=0.5,
            random_seed=seed,
            verbose=False,
            thread_count=-1,
        )
        model.fit(train, target, cat_features=cat_indices, verbose=False)
        predictions.append(model.predict(x_pred))
    return np.mean(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    data = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train_test.csv")
    validation = pd.read_csv(data / "validation.csv")
    december = pd.read_csv(data / "december_chart_inputs.csv")

    x_train = add_features(train)
    x_val = add_features(validation)

    # 1) Main model: predict sqrt(rate), then square back.
    cat_cols = [c for c in x_train.columns if x_train[c].dtype == "object"]
    cat_indices = [x_train.columns.get_loc(c) for c in cat_cols]

    sqrt_pred = fit_and_predict(
        x_train, np.sqrt(train["posted_rate"]), x_val, cat_indices,
        iterations=260
    )
    sqrt_pred = np.maximum(sqrt_pred, 0) ** 2

    # 2) Complementary model: predict rate per mile, then multiply by distance.
    rpm = train["posted_rate"] / train["distance"]
    rpm_pred = fit_and_predict(
        x_train, rpm, x_val, cat_indices,
        iterations=150
    )
    rpm_pred = rpm_pred * validation["distance"].to_numpy()

    final_pred = 0.75 * sqrt_pred + 0.25 * rpm_pred
    final_pred = np.maximum(final_pred, 1e-6)

    pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": final_pred,
    }).to_csv(out / "validation_predictions.csv", index=False)

    # December chart inputs have fewer columns. Recover coordinates from city names.
    dec_full = add_missing_december_columns(december, pd.concat([train, validation], ignore_index=True))
    x_dec = add_features(dec_full)[x_train.columns]

    sqrt_dec = fit_and_predict(
        x_train, np.sqrt(train["posted_rate"]), x_dec, cat_indices,
        iterations=260
    )
    sqrt_dec = np.maximum(sqrt_dec, 0) ** 2

    rpm_dec = fit_and_predict(
        x_train, rpm, x_dec, cat_indices,
        iterations=150
    )
    rpm_dec = rpm_dec * december["distance"].to_numpy()

    dec_pred = np.maximum(0.75 * sqrt_dec + 0.25 * rpm_dec, 1e-6)
    december["predicted_rate"] = dec_pred
    december.to_csv(data / "december_chart_inputs.csv", index=False)

    print("Wrote validation_predictions.csv and completed data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
