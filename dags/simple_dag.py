"""DAG: Simple DAG
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

def print_hello():
    print("Hello world!")

with DAG(
    dag_id="simple_dag",
    default_args=default_args,
    description="Simple DAG",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["simple"],
) as dag:
    print_hello_message = PythonOperator(
        task_id="print_hello_id",
        python_callable=print_hello,
    )

    print_hello_message
