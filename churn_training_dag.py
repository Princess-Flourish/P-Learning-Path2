from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# DAG definition
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="churn_model_training_dag",
    default_args=default_args,
    description="Train churn prediction model daily",
    schedule="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    train_model = BashOperator(
        task_id="train_model_task",
        bash_command="python C:\\Users\\user\\Documents\\Bluechip Technologies\\Mlops_demo_project\\scripts\\train_model.py"
    )

    run_inference = BashOperator(
        task_id="run_inference_task",
        bash_command="python C:\\Users\\user\\Documents\\Bluechip Technologies\\Mlops_demo_project\\scripts\\inference.py"
    )

    train_model >> run_inference