import airflow
from airflow import DAG
from airflow.operators import BashOperator
from datetime import datetime, timedelta
from airflow.contrib.operators.databricks_operator import DatabricksSubmitRunOperator
import papermill as pm
from airflow.operators import PythonOperator
from airflow.operators.papermill_operator import PapermillOperator

today_date = datetime.today()


default_args = {
   'owner': 'airflow',
   'depends_on_past': False,
   'start_date': today_date,
   'retries': 1,
   'retry_delay': timedelta(minutes=5),
}

dag = DAG('databricks', default_args=default_args, schedule_interval= '@daily')

new_cluster = {
    'spark_version': '6.2.x-scala2.11',
    'node_type_id': 'Standard_DS3_v2',
    'num_workers': 2,
    #'libraries':  [{
    #    'library': 'azure'
    #}]
}

notebook_task_params = {
    #'new_cluster': new_cluster,
    'existing_cluster_id' : '0227-193205-rifer660',
    'notebook_task': {
        'notebook_path': '/Users/tnormile@e2evapoutlook.onmicrosoft.com/Model2',
    }
}

notebook_task_two_params = {
    'new_cluster': new_cluster,
    #'existing_cluster_id' : '0227-193205-rifer660',
    'notebook_task': {
        'notebook_path': '/Users/tnormile@e2evapoutlook.onmicrosoft.com/helloworld',
    }
}

notebook_task = DatabricksSubmitRunOperator(
    task_id='notebook_task',
    dag=dag,
    json=notebook_task_params)

notebook_two_task = DatabricksSubmitRunOperator(
    task_id='notebook_two_task',
    dag=dag,
    json=notebook_task_two_params)

# Example of using the named parameters of DatabricksSubmitRunOperator
# to initialize the operator.

#run_this = PapermillOperator(
    #task_id='run_jupyter_notebook',
    #dag=dag,
    #input_nb="/home/tnormile/helloworld.ipynb",
    #output_nb="/home/tnormile/out-{{ execution_date }}.ipynb",
    #parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"}
#)

dbtask = BashOperator(
    task_id='dbjob',
    depends_on_past=False,
    bash_command='curl -X POST -u tnormile@e2evapoutlook.onmicrosoft.com:Colleen123! https://eastus.azuredatabricks.net/api/2.0/jobs/run-now -d \'{\"job_id\":}\'',
    dag=dag)

notebook_task > notebook_two_task
