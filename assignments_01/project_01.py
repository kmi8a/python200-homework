import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from prefect import task, flow, get_run_logger

OUTPUT = Path('outputs/')

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    
    base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/"
    years = range(2015, 2025) 
    df_list = []

    # standardization of column names
    rename_map = {
        'Happiness score': 'happiness_score', 'Ladder score': 'happiness_score',
        'Regional indicator': 'region',
        'GDP per capita': 'gdp_per_capita',
        'Social support': 'social_support',
        'Healthy life expectancy': 'healthy_life_expectancy',
        'Freedom to make life choices': 'freedom',
        'Generosity': 'generosity',
        'Perceptions of corruption': 'corruption'
    }

    # define columns to be treated as numeric
    numeric_cols = [
        'happiness_score', 'gdp_per_capita', 'social_support', 
        'healthy_life_expectancy', 'freedom', 'generosity', 'corruption'
    ]

    for year in years:
        url = f"{base_url}world_happiness_{year}.csv"
        df = pd.read_csv(url, sep=';', on_bad_lines='warn')
        
        df.columns = df.columns.str.strip()
        df.rename(columns=rename_map, inplace=True)
        
        # clean 'region' column
        if 'region' in df.columns:
            df['region'] = df['region'].str.strip()
            
        # convert columns to numeric
        for col in numeric_cols:
            if col in df.columns:
                # replace comma with dot
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['year'] = year
        df_list.append(df)
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # save csv to outputs folder
    combined_df.to_csv(OUTPUT / "merged_happiness.csv", index=False)
    logger.info(f"Successfully merged {len(combined_df)} rows from {len(years)} years.")
    return combined_df

@task
def compute_statistics(df):
    logger = get_run_logger()

    # extract mean, median and standard deviation
    mean_val = df['happiness_score'].mean()
    median_val = df['happiness_score'].median()
    std_val = df['happiness_score'].std()
    
    logger.info(f"Overall Happiness Stats: Mean={mean_val:.2f}, Median={median_val:.2f}, Std={std_val:.2f}")
    
    # regional breakdown
    if 'region' in df.columns:
        regional_mean = df.groupby('region')['happiness_score'].mean()
        logger.info(f"Regional Mean Happiness: {regional_mean.to_dict()}")
    else:
        logger.warning("Column 'region' not found in DataFrame.")

@task
def generate_visuals(df):
    logger = get_run_logger()

    # remove rows where happiness_score or year is missing to prevents the plotting from receiving empty data
    plot_df = df.dropna(subset=['happiness_score', 'year'])
    
    # histogram
    plt.figure()
    sns.histplot(plot_df['happiness_score'])
    plt.savefig(OUTPUT / "happiness_histogram.png")
    
    # boxplot
    plt.figure()
    sns.boxplot(data=plot_df, x='year', y='happiness_score')
    plt.savefig(OUTPUT / "happiness_by_year.png")
    
    
    # scatter
    plt.figure()
    sns.scatterplot(data=df, x='gdp_per_capita', y='happiness_score')
    plt.savefig(OUTPUT / "gdp_vs_happiness.png")
    logger.info("Saved gdp_vs_happiness.png")
    
    # heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True)
    plt.savefig(OUTPUT / "correlation_heatmap.png")
    logger.info("Saved correlation_heatmap.png")

@task
def run_hypothesis_testing(df):
    logger = get_run_logger()
    df_2019 = df[df['year'] == 2019]['happiness_score'].dropna()
    df_2020 = df[df['year'] == 2020]['happiness_score'].dropna()
    
    t_stat, p_val = stats.ttest_ind(df_2019, df_2020)
    
    logger.info(f"2019 vs 2020 T-Test: t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        logger.info("Interpretation: There is a statistically significant difference in happiness scores before and during the start of the pandemic.")
    else:
        logger.info("Interpretation: No statistically significant difference found between 2019 and 2020 happiness scores.")

    return t_stat, p_val

@task
def perform_correlation_analysis(df):
    logger = get_run_logger()
    numeric_cols = df.select_dtypes(include=['number']).columns.drop(['year', 'happiness_score'])
    results = []
    
    for col in numeric_cols:
        corr, p = stats.pearsonr(df[col], df['happiness_score'])
        results.append((col, corr, p))
    
    num_tests = len(results)
    adj_alpha = 0.05 / num_tests

    strongest_var, strongest_corr = None, 0
    
    for col, corr, p in results:
        sig_standard = "Significant" if p < 0.05 else "Not Significant"
        sig_adj = "Significant" if p < adj_alpha else "Not Significant"
        logger.info(f"Var: {col} | Corr: {corr:.2f} | Standard: {sig_standard} | Bonferroni: {sig_adj}")

        if p < adj_alpha and abs(corr) > abs(strongest_corr):
            strongest_var, strongest_corr = col, corr

    return strongest_var, strongest_corr

@task
def generate_summary_report(df, ttest_results, correlation_results):
    logger = get_run_logger()
    t_stat, p_val = ttest_results
    strongest_var, strongest_corr = correlation_results
    
    logger.info("===== SUMMARY =====")
    
    # Total number of countries and years in the merged dataset.
    num_countries = df['Country'].nunique()
    num_years = df['year'].nunique()
    logger.info(f'Dataset covers {num_countries} countries across {num_years} years.')
    
    # The top 3 and bottom 3 regions by mean happiness score.
    if 'region' in df.columns:
        regional_mean = df.groupby('region')['happiness_score'].mean().sort_values(ascending=False)
        top_3 = ', '.join(f"{r} ({v:.2f})" for r, v in regional_mean.head(3).items())
        bottom_3 = ', '.join(f'{r} ({v:.2f})' for r, v in regional_mean.tail(3).sort_values().items())
        logger.info(f'Top 3 happiest regions: {top_3}')
        logger.info(f'Bottom 3 happiest regions: {bottom_3}')
    else:
        logger.warning('Region column not found; skipping regional ranking in summary.')
    
    # The result of the pre/post-2020 t-test in plain language.
    if pd.notna(p_val):
        if p_val < 0.05:
            logger.info(f'There was a statistically significant change in happiness scores from 2019 to 2020 (p={p_val:.4f}), consistent with a pandemic-era shift.')
        else:
            logger.info(f'No statistically significant change in happiness scores was found between 2019 and 2020 (p={p_val:.4f}).')
    else:
        logger.warning('T-test result unavailable (likely missing data for 2019 or 2020).')
    
    # The variable most strongly correlated with happiness score (after Bonferroni correction).
    if strongest_var:
        direction = 'positively' if strongest_corr > 0 else 'negatively'
        logger.info(f"'{strongest_var}' is the variable most strongly correlated with happiness after Bonferroni correction (r={strongest_corr:.2f}, {direction} correlated).")
    else:
        logger.info('No variable remained significantly correlated with happiness after Bonferroni correction.')
    
    logger.info("===== END SUMMARY =====")

@flow
def happiness_pipeline():
    df = load_and_merge_data()
    compute_statistics(df)
    generate_visuals(df)
    ttest_results = run_hypothesis_testing(df)
    correlation_results = perform_correlation_analysis(df)
    generate_summary_report(df, ttest_results, correlation_results)

if __name__ == '__main__':
    happiness_pipeline()
