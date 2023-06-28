ARG WORKDIR=/app

FROM svizor/zoomcamp-model:mlops-3.10.0-slim as builder
ARG WORKDIR
WORKDIR $WORKDIR

COPY poetry.lock .
COPY pyproject.toml .

RUN pip install -U pip poetry
RUN poetry config virtualenvs.create false

RUN poetry install --only main
# cleanup
RUN rm poetry.lock
RUN poetry cache clear pypi --all
RUN pip uninstall -y poetry

FROM svizor/zoomcamp-model:mlops-3.10.0-slim as runtime
ARG WORKDIR
WORKDIR $WORKDIR

COPY --from=builder / /

COPY ./starter.py $WORKDIR

CMD [ "python", "starter.py", "--year", "2022", "--month", "4" ]
