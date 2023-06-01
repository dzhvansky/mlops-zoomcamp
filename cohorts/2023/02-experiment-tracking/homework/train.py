import os
import pickle
import typing

import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


MLFLOW_TRACKING_URI: typing.Final[str] = "sqlite:///mlflow.db"
EXPERIMENT_NAME: typing.Final[str] = "nyc-taxi-experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option("--data_path", default="./output", help="Location where the processed NYC taxi trip data was saved")
@click.option("--run_name", default=None, help="Location where the processed NYC taxi trip data was saved")
def run_train(data_path: str, run_name: typing.Optional[str]):
    with mlflow.start_run(run_name=run_name):

        train_path: str = os.path.join(data_path, "train.pkl")
        val_path: str = os.path.join(data_path, "val.pkl")

        mlflow.log_param("train-data-path", train_path)
        mlflow.log_param("valid-data-path", val_path)

        X_train, y_train = load_pickle(train_path)
        X_val, y_val = load_pickle(val_path)

        rf_params: dict[str, int] = {"max_depth": 10, "random_state": 0}

        for param, value in rf_params.items():
            mlflow.log_param(param, value)

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
