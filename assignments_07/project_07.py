import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dotenv import load_dotenv
from smolagents import OpenAIServerModel, CodeAgent, tool
from scipy.stats import pearsonr

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

api_key = os.getenv("OPENAI_API_KEY")

DATA_DIR = os.path.join(os.getcwd(), "resources")
DATA_PATH = os.path.join(DATA_DIR, "merged_happiness.csv")

os.makedirs("outputs", exist_ok=True)

df= None

# Task 1: Define Your Tools

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory, update the global dataframe 'df', 
    and return a dictionary containing its shape and column names.

    This function attempts to load a pre-merged CSV file from DATA_PATH. If that
    file does not exist, it falls back to loading and merging all yearly CSV files
    found in the resources directory (DATA_DIR).

    NOTE: This function updates the global variable `df`. Do NOT pass the return value 
    of this function into `pd.DataFrame()`. Use the global variable `df` for data analysis.

    Returns:
        dict: A dictionary containing 'shape' and 'columns', or an error dictionary if
        loading fails.
    """
    global df
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            if not os.path.exists(DATA_DIR):
                return {"error": f"Directory {DATA_DIR} does not exist."}
            
            csv_files = [
                os.path.join(DATA_DIR, f)
                for f in os.listdir(DATA_DIR)
                if f.endswith('.csv') and f != 'merged_happiness.csv'
            ]
            
            if not csv_files:
                return {"error": f"No yearly CSV files found in {DATA_DIR}."}
            
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)

        return {
            "shape": df.shape,
            "columns": list(df.columns)
        }
    except Exception as e:
        return {"error": f"Critical error loading data: {str(e)}"}


@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column (str): The name of the column to summarize.

    Returns:
        dict: A dictionary of descriptive statistics from pandas describe(), or an error dictionary.
    """
    global df
    if df is None:
        return {"error": "Dataset not loaded. Please call load_happiness_data first."}
    try:
        if column not in df.columns:
            return {"error": f"Column '{column}' not found."}
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not numeric."}
            
        return df[column].describe().to_dict()
    except Exception as e:
        return {"error": str(e)}


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1 (str): The first numeric column.
        col2 (str): The second numeric column.

    Returns:
        dict: Dictionary containing pearson_r and p_value, or an error dictionary.
    """
    global df
    if df is None:
        return {"error": "Dataset not loaded. Please call load_happiness_data first."}
    try:
        if col1 not in df.columns or col2 not in df.columns:
            return {"error": "Invalid columns or dataset."}
        
        valid_data = df[[col1, col2]].dropna()
        r, p = pearsonr(valid_data[col1], valid_data[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list | dict:
    """
    Return the top N countries ranked by a given column for a specific year as a list of dicts.

    Args:
        column (str): Column to rank by.
        year (int): Target year.
        n (int): Number of top countries.

    Returns:
        list | dict: Top countries list of dictionaries, each with 'country' and the requested column value, or an error dictionary.
    """
    global df
    if df is None:
        return {"error": "Dataset not loaded. Please call load_happiness_data first."}
    try:
        if 'year' not in df.columns or 'Country' not in df.columns or column not in df.columns:
            return {"error": "Required columns ('year', 'Country', or requested column) not found in dataset."}
        
        filtered = df[df['year'] == year]
        if filtered.empty:
            return {"error": f"No data found for year {year}."}
        
        sorted_df = filtered.sort_values(by=column, ascending=False).head(n)
        
        result_list = [
            {
                "country": row['Country'], 
                column: row[column]
            } 
            for _, row in sorted_df.iterrows()
        ]
        
        return result_list
    except Exception as e:
        return {"error": str(e)}


# Task 2: Build the Agent

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
- You are a data analyst assistant for the World Happiness dataset.
- Use the available tools for loading data, summarizing columns, computing correlations, and ranking countries.
- CRITICAL: `load_happiness_data()` returns a metadata dictionary with 'shape' and 'columns'. When you write custom Python code that requires direct pandas operations, 
  load the dataset locally inside your code block using `df = pd.read_csv(DATA_PATH)` or `pd.read_csv("resources/merged_happiness.csv")` since the agent's execution 
  sandbox runs in an isolated scope.
- PLOTTING RULE: When creating multi-category charts, use a wide figure size (e.g., `figsize=(14, 7)`), place the legend outside using `plt.legend(title='Region', bbox_to_anchor=(1.02, 1), loc='upper left')`, and add `plt.subplots_adjust(right=0.75)` before saving to prevent the chart area from looking squished.
- ALWAYS save generated plots using `plt.savefig(output_dir + 'filename.png')` instead of using `plt.show()`, since this runs in a headless environment.
- Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy", "scipy.stats", "os"],
    max_steps=8,
)

if __name__ == "__main__":

# Task 3: Run Guided Queries

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        try:
            response = agent.run(query)
            print(f"\nAgent Response:\n{response}")
        except Exception as e:
            print(f"An error occurred during execution: {e}")


# Task 4: Your Own Questions

    my_questions = [
        "What is the statistical correlation between social support and healthy life expectancy, and is it significant?",
        "Generate a scatter plot showing the relationship between GDP per capita and happiness score for the year 2022, include a fitted linear regression trendline, Save the plot to outputs/gdp_happiness_scatter.png."
    ]

    for query in my_questions:
        print(f"\n--- Query: {query} ---")
        try:
            response = agent.run(query)
            print(f"\nAgent Response:\n{response}")
        except Exception as e:
            print(f"An error occurred during execution: {e}")


# Task 5: Reflection

# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
# A: The agent evaluated the p-value returned by `compute_correlation` and compared it against the standard alpha threshold of 0.05.
#    It correctly reported whether the p-value fell below this threshold to establish statistical significance.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than you expected, or less? Describe one specific example.
# A: The agent's capability to switch between predetermined tools and execution of new code without breaking execution flow
#    was quite surprising, it demonstrated solid 'reasoning', a specific example is the plot 
#
# 3. What one additional tool would make this agent meaningfully more useful? 
#    Describe what it would do and what kind of question it would help the agent answer. (You do not need to implement it.)
# A: a tool that makes projections to the future based on the available data, it would be helpful to estimate were a country is heading, and also
#    would help the agent answer questions such as: How much would a 15% increase in social support impact a country's predicted happiness score over the next three years?