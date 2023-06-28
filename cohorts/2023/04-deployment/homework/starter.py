#!/usr/bin/env python
# coding: utf-8

import pathlib
import pickle

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer


BASE_URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"

OUTPUT_DIR: str = "output/yellow"

CATEGORICAL: list[str] = ['PULocationID', 'DOLocationID']


def _data_slice_id(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def load_model() -> tuple[DictVectorizer, RandomForestRegressor]:
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model


def read_data(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df


def predict(df: pd.DataFrame, dv: DictVectorizer, model: RandomForestRegressor) -> np.ndarray:
    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def save_results(df_result: pd.DataFrame, output_dir: str, year: int, month: int) -> None:
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file: str = f"./{output_dir}/{_data_slice_id(year=year, month=month)}.parquet"
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


@click.command()
@click.option("--year", type=int, help="Year of data slice.")
@click.option("--month", type=int, help="Year of data slice.")
def run(year: int, month: int) -> None:
    dv, model = load_model()

    df = read_data(f'{BASE_URL}{_data_slice_id(year=year, month=month)}.parquet')
    y_pred = predict(df, dv=dv, model=model)
    print(f"Mean trip duration for {year=}:{month=} = {y_pred.mean()}")

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()

    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    save_results(df_result, output_dir=OUTPUT_DIR, year=year, month=month)


if __name__ == "__main__":
    run()
