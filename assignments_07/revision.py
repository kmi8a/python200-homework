# Issues that need revision
# 1) project_07.py does not follow the required DATA_PATH / fallback loading setup
# The instructions say to load assignments_01/outputs/merged_happiness.csv if it exists, otherwise fall back to the yearly CSVs
# in assignments/resources/happiness_project/. Your code points DATA_PATH to resources/merged_happiness.csv and falls back to the local
# resources/ folder instead. That means the project is not using the required data path structure.

# 2) load_happiness_data does not update the shared global df
# The assignment explicitly says to define df = None at the top and update it inside load_happiness_data. In your tool, df is assigned locally inside the function, so the shared dataset state is not preserved for the other tools.

# 3) The project tools do not consistently use the shared loaded dataset
# summarize_column, compute_correlation, and get_top_n_countries all read from DATA_PATH directly instead of using the shared global df that should be populated by load_happiness_data. This breaks the intended conversational agent flow, where later queries depend on the loaded dataset remaining in memory.

# 4) get_top_n_countries does not match the required return shape
# The instructions ask for the top n rows as a list of dicts, each with 'country' and the requested column value. Your tool returns {'top_countries': result_list} instead of returning the list directly.

# 5) warmup_07.py is missing the required Q4–Q9 deliverable structure
# The warmup file includes a lot of work, but it does not clearly satisfy the assignment’s required organization for the later questions. In particular, the required comments and outputs for the smolagents section should be easy to identify and aligned with the prompt-specific tasks.

# 6) Q2 prediction comment is incomplete
# The instructions ask for a prediction about whether run_agent('Convert 100 degrees Celsius to Fahrenheit') will trigger a tool call and how many API calls will be made. You answered that, but the explanation is a bit too brief to clearly show the reasoning expected from the lesson.

# Bottom line
# Because the project’s data-loading flow and shared state do not match the instructions, I need to mark this as requiring revision. The strongest next step is to align the project with the exact DATA_PATH/fallback requirements and make all tools operate on the same loaded df.