# Homework №3 solution

- Run prefect server
```bash
poetry run prefect server start
```

```bash
poetry shell
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
prefect project init
prefect deploy --cron "0 9 3 * *" 03-orchestration/3.4/orchestrate.py:main_flow -p mlops-pool
prefect worker start --pool 'mlops-pool'
prefect block register -m prefect_email
python 03-orchestration/3.4/create_email_block.py
prefect deploy 03-orchestration/3.4/orchestrate.py:main_flow -p mlops-pool
```