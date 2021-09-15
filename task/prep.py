import prefect
from prefect import task, Parameter


@task
def hello_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Hello {name}!")

    return name


@task
def hi_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Hi {name}!")

    return name


@task
def buy_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Buy {name}!")
