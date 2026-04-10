import pandas as pd
from sklearn import datasets
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import CloudWorkspace

def create_project(project_name,
    project_description,
    evidently_api_key,
    evidently_org_id):
    ws = CloudWorkspace(token=evidently_api_key, url="https://app.evidently.cloud")
    project = ws.create_project(project_name, org_id=evidently_org_id)
    project.description = project_description
    project.save()
    return project

def create_schema(numerical_columns=[], categorical_columns=[]):
    schema = DataDefinition(
    numerical_columns=numerical_columns,
    categorical_columns=categorical_columns,
    )
    return schema

def generate_drift_monitoring_report(reference_dataset,
    production_dataset,
    project,
    schema
    ):
    eval_data_1 = Dataset.from_pandas(
    pd.DataFrame(production_dataset),
    data_definition=schema
    )
    eval_data_2 = Dataset.from_pandas(
    pd.DataFrame(reference_dataset),
    data_definition=schema
    )
    report = Report([
    DataDriftPreset(drift_share=0.7, method="psi")
    ])
    my_eval = report.run(eval_data_1, eval_data_2)
    ws.add_run(project.id, my_eval, include_data=False)

    
