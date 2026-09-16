from prefect import flow, task
from prefect.logging import get_run_logger

# --- Prefect Orchestration ---

# Q1
# In a comment block, answer: what is the difference between a @task and a @flow in Prefect?
# You have a helper function that converts a temperature from Celsius to Fahrenheit — a pure, in-memory calculation with no I/O.
# Would you decorate it with @task? Why or why not?

# Answer
# A @flow orchestrates and contains the entire workflow, it calls the different @tasks in a determined order and manages the run as a whole,
# a @task is then the single unit of work.
# I wouldnt decorate a helper function that converts a temperature from Celsius to Farenheit with @task, as doing so introduces runtime
# overhead, if the operation is a pure, lightning-fast, in-memory calculation with zero I/O it is then more efficient to call it directly.

# Q2
# Write just the decorator line for a task named call_api that retries up to 3 times with a 30-second delay between attempts.

@task(retries=3, retry_delay_seconds=30)

# Q3
# You run your pipeline and the Prefect UI shows: extract is Completed, load_raw is Completed, transform is Failed, load_enriched never ran. 
# In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific information would you expect to find there?

# Answer
# In the UI, inside the "Runs" tab, after clicking on the failed run, you click on the failed task block ('transform' in this case).
# there you will find the Python exception, traceback and error message thown by the failed task.

# --- Production Patterns ---

# Q1
# Production Question 1
# In a comment block, explain what raise_for_status() does and why it is better than writing if response.status_code != 200: print("error") in a pipeline task.
# What happens to downstream tasks in each case when the API returns a 500 error?

# Answer
# raise_for_status() will propagate the exception inmediatly causing prefect to mark the task as failed, as a consequence of this the error is then shown on the
# logs and the following downstream task will not run, it is better that writing if response.status_code != 200: print("error") because when any other type of error happens,
# it is catched by the exception instead of silently failing, compromising the following steps in the pipeline.

# Q2
# Your load_raw task uses upsert with on_conflict="date" instead of insert. The pipeline crashes halfway through the transform step. 
# You fix the bug and re-run from the beginning. In a comment block, explain: what does upsert protect you from in this scenario, and what would happen
# if you had used plain insert instead?

# Answer
# In this scenario, upsert ensures idempotency, protecting against the writing of records that conflict on the 'date' column.
# using a plain insert would try to write duplicate rows for dates that were already processed during the 
# first partial run of the script, this would cause a unique constraint violation in the database, that in turn causes the script to crash. 

# Q3
# Write a task stub — just the function signature, decorator, and a single log line — that uses get_run_logger() to log an INFO message saying how many
# enrichment records were upserted. The function should accept enrichment_records (a list) as its argument.

@task(retries=2, retry_delay_seconds=5)
def info_upserted_records(enrichment_records: list) -> None:
    logger = get_run_logger()
    logger.info(f'{len(enrichment_records)} enrichment records were upserted')


# Answer

# Q4
# In a comment block, explain how the incremental processing check in the transform task contributes to idempotency. If you removed it and the pipeline
# ran the ML and LLM steps on all 365 records every time, what would be the practical consequences (in terms of cost, time, and data correctness)?

# Answer
# The check queries the database for existing dates and filters `raw_records` to select only whatever records are missing, this ensures that re-running
# the task skips the records previously processed. removing this would have significant consequences on costs, as processing all the records everytime the script
# runs would trigger hundreds of calls to the OpenAI API, driving up the costs unnecessarily. The time spent on the network requests for every record would rise
# in a significant manner turning a fast incremental run into a slow bottleneck. Lastly but not less important would be the consequence that hitting external APIs
# for hundreds of records increases exposure to rate limits, timeouts, and transient errors that could crash the pipeline.