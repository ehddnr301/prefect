from Pipeline import Pipeline
from prefect import Client
from prefect.schedules.clocks import CronClock


if __name__ == '__main__':
    Client().create_project(project_name='test')
    pipeline = Pipeline("test", "test_flow")
    pipeline.create_flow()

