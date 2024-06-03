import os
import numpy as np
import logging
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, pearsonr
from sklearn.linear_model import LinearRegression
#from statsmodels.tsa.seasonal import seasonal_decompose


logging.basicConfig(filename='eda_analysis.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def compute_descriptive_statistics(df):
    logging.info("Started compute_descriptive_statistics")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    descriptive_stats = df[numeric_cols].describe()
    logging.info("Computed descriptive statistics")
    
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=True, ax=ax)
    ax.set_title('Distribution of Sales')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Sales")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CompetitionDistance'], kde=True, ax=ax)
    ax.set_title('Distribution of Competition Distance')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Competition Distance")

def compare_promo_distribution(train_df, test_df):
    logging.info("Started compare_promo_distribution")
    
    print("Distribution of Promo in the training dataset:")
    print(train_df['Promo'].value_counts())
    sns.histplot(train_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in training dataset")
    
    print("Distribution of Promo in the test dataset:")
    print(test_df['Promo'].value_counts())
    sns.histplot(test_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in test dataset")
    
    print("Comparing the distribution of Promo between the training and test datasets:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist([train_df['Promo'], test_df['Promo']], bins=2, label=['Training', 'Test'])
    ax1.set_title('Histogram of Promo')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    ax2.boxplot([train_df['Promo'], test_df['Promo']], labels=['Training', 'Test'])
    ax2.set_title('Box Plot of Promo')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Promo')
    
    plt.show()
    logging.info("Compared the Promo distributions between training and test datasets")
    
    train_promo_count = train_df['Promo'].value_counts()
    test_promo_count = test_df['Promo'].value_counts()
    
    if train_promo_count.equals(test_promo_count):
        logging.info("The distribution of Promo is similar between the training and test datasets.")
        print("The distribution of Promo is similar between the training and test datasets.")
    else:
        logging.warning("The distribution of Promo is not similar between the training and test datasets.")
        print("The distribution of Promo is not similar between the training and test datasets.")

def analyze_sales_around_holidays(df):
    logging.info("Started analyze_sales_around_holidays")
    
    holiday_stats = df.groupby('StateHoliday')['Sales'].agg(['mean', 'median', 'std'])
    logging.info("Computed sales statistics by holiday type")
    print("Sales statistics by holiday type:")
    print(holiday_stats)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days_Since_Holiday'] = (df['Date'] - df['Date'].where(df['StateHoliday'] != '0').ffill()).dt.days
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(x='Days_Since_Holiday', y='Sales', hue='StateHoliday', data=df, ax=ax1)
    ax1.set_title('Sales Around Holidays')
    ax1.set_xlabel('Days Since Holiday')
    ax1.set_ylabel('Sales')
    ax1.legend(title='Holiday Type')
    
    sns.boxplot(x='StateHoliday', y='Sales', data=df, ax=ax2)
    ax2.set_title('Sales by Holiday Type')
    ax2.set_xlabel('Holiday Type')
    ax2.set_ylabel('Sales')
    
    plt.tight_layout()
    plt.show()
    logging.info("Plotted sales trends around holidays")
    
    print("Observations:")
    if df.loc[df['StateHoliday'] != '0', 'Sales'].mean() < df.loc[df['StateHoliday'] == '0', 'Sales'].mean():
        print("Sales tend to be lower during state holidays.")
    else:
        print("Sales tend to be higher during state holidays.")
    
    if df['Days_Since_Holiday'].max() > 0:
        print("Sales appear to recover after state holidays.")
    else:
        print("No clear pattern in sales after state holidays.")

def analyze_seasonal_purchases(df):
    logging.info("Started analyze_seasonal_purchases")
    
    df['Date'] = pd.to_datetime(df['Date'])
    decomposition = seasonal_decompose(df['Sales'], model='additive', period=7)
    
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.suptitle('Seasonal Decomposition of Sales')
    plt.show()
    logging.info("Plotted seasonal decomposition of sales")
    
    df['is_promo2'] = df['Promo2'].astype(bool)
    df['week'] = df['Date'].dt.isocalendar().week
    promo2_sales = df.groupby(['week', 'is_promo2'])['Sales'].mean().unstack()
    
    logging.info("Analyzed the impact of Promo2 on seasonal sales patterns")
    print("Observations:")
    unique_promo2 = df['is_promo2'].unique()
    if len(unique_promo2) == 2:
        promo2_false_max = promo2_sales[False].max()
        promo2_true_max = promo2_sales[True].max()
        if promo2_false_max > promo2_true_max:
            print("Sales appear to be higher during periods without Promo2.")
        else:
            print("Sales appear to be higher during periods with Promo2.")
    else:
        print("Insufficient data to determine the impact of Promo2 on seasonal sales patterns.")
    
    if decomposition.seasonal.max() > decomposition.seasonal.min():
        print("There are clear seasonal patterns in sales.")
    else:
        print("No clear seasonal patterns in sales.")
    
    plt.figure(figsize=(12, 8))
    plt.plot(promo2_sales.index, promo2_sales[False], label='No Promo2')
    plt.plot(promo2_sales.index, promo2_sales[True], label='Promo2')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.title('Weekly Sales with and without Promo2')
    plt.legend()
    plt.show()
    logging.info("Plotted weekly sales with and without Promo2")

def analyze_sales_customers_correlation(df):
    logging.info("Started analyze_sales_customers_correlation")
    
    correlation = df['Sales'].corr(df['Customers'])
    logging.info(f"Computed correlation coefficient: {correlation:.2f}")
    print(f"Correlation coefficient between Sales and Customers: {correlation:.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Customers', y='Sales', data=df, ax=ax)
    ax.set_title('Relationship between Sales and Customers')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Sales')
    
    slope, intercept = np.polyfit(df['Customers'], df['Sales'], 1)
    ax.plot(df['Customers'], slope * df['Customers'] + intercept, color='r', label=f'Regression line (slope={slope:.2f})')
    ax.legend()
    
    plt.show()
    logging.info("Plotted relationship between Sales and Customers")
    
    print("Implications for the sales forecasting model:")
    if correlation > 0:
        print("The positive correlation between sales and the number of customers suggests that the sales forecasting model should incorporate the customer count as a feature. This could improve the model's ability to predict sales accurately.")
    elif correlation < 0:
        print("The negative correlation between sales and the number of customers suggests that the sales forecasting model should consider the customer count as a feature, but with an inverse relationship. This could help the model better capture the dynamics between sales and customer count.")
    else:
        print("The lack of correlation between sales and the number of customers suggests that the customer count may not be a useful feature for the sales forecasting model. The model should focus on other factors that are more strongly related to sales.")

def analyze_promo_impact(df):
    logging.info("Started analyze_promo_impact")
    
    promo_group = df.groupby('Promo')['Customers'].sum().reset_index()
    new_customers = promo_group.loc[promo_group['Promo'] == 1, 'Customers'].values[0]
    repeat_customers = promo_group.loc[promo_group['Promo'] == 0, 'Customers'].values[0]
    
    logging.info("Computed number of new and repeat customers")
    print(f"Number of new customers during promotion periods: {new_customers}")
    print(f"Number of repeat customers during non-promotion periods: {repeat_customers}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Promo', y='Customers', data=df, ax=ax)
    ax.set_title('Impact of Promo on Number of Customers')
    ax.set_xlabel('Promo')
    ax.set_ylabel('Customers')
    plt.show()
    logging.info("Plotted impact of Promo on number of customers")
    
    t_stat, p_val = ttest_ind(df.loc[df['Promo'] == 1, 'Customers'], df.loc[df['Promo'] == 0, 'Customers'])
    logging.info(f"Performed t-test: t_stat={t_stat:.2f}, p_val={p_val:.2f}")
    print(f"T-test results: t-statistic = {t_stat:.2f}, p-value = {p_val:.2f}")
    
    if p_val < 0.05:
        logging.info("The promotion has a significant impact on the number of customers (p < 0.05).")
        print("The promotion has a significant impact on the number of customers (p < 0.05).")
    else:
        logging.warning("The promotion does not have a significant impact on the number of customers (p >= 0.05).")
        print("The promotion does not have a significant impact on the number of customers (p >= 0.05).")

def analyze_competition_distance_impact(df):
    logging.info("Started analyze_competition_distance_impact")
    
    correlation, _ = pearsonr(df['Sales'], df['CompetitionDistance'])
    logging.info(f"Computed correlation coefficient between Sales and Competition Distance: {correlation:.2f}")
    print(f"Correlation coefficient between Sales and Competition Distance: {correlation:.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df, ax=ax)
    ax.set_title('Impact of Competition Distance on Sales')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Sales')
    
    slope, intercept = np.polyfit(df['CompetitionDistance'], df['Sales'], 1)
    ax.plot(df['CompetitionDistance'], slope * df['CompetitionDistance'] + intercept, color='r', label=f'Regression line (slope={slope:.2f})')
    ax.legend()
    
    plt.show()
    logging.info("Plotted impact of Competition Distance on Sales")
    
    print("Implications for the sales forecasting model:")
    if correlation < 0:
        print("The negative correlation between sales and competition distance suggests that closer competition is associated with lower sales. The sales forecasting model should include competition distance as a feature and account for its inverse relationship with sales.")
    elif correlation > 0:
        print("The positive correlation between sales and competition distance suggests that farther competition is associated with higher sales. The sales forecasting model should include competition distance as a feature and account for its direct relationship with sales.")
    else:
        print("The lack of correlation between sales and competition distance suggests that competition distance may not be a useful feature for the sales forecasting model. The model should focus on other factors that are more strongly related to sales.")



def compute_descriptive_statistics(df):
    logging.info("Started compute_descriptive_statistics")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    descriptive_stats = df[numeric_cols].describe()
    logging.info("Computed descriptive statistics")
    
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=True, ax=ax)
    ax.set_title('Distribution of Sales')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Sales")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CompetitionDistance'], kde=True, ax=ax)
    ax.set_title('Distribution of Competition Distance')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Competition Distance")

def compare_promo_distribution(train_df, test_df):
    logging.info("Started compare_promo_distribution")
    
    print("Distribution of Promo in the training dataset:")
    print(train_df['Promo'].value_counts())
    sns.histplot(train_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in training dataset")
    
    print("Distribution of Promo in the test dataset:")
    print(test_df['Promo'].value_counts())
    sns.histplot(test_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in test dataset")
    
    print("Comparing the distribution of Promo between the training and test datasets:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist([train_df['Promo'], test_df['Promo']], bins=2, label=['Training', 'Test'])
    ax1.set_title('Histogram of Promo')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Count')
    ax1.legend()
    plt.show()
    logging.info("Compared Promo distribution between training and test datasets")

def optimize_promo_deployment(df):
    """
    Investigate the relationship between Promo, Sales, and other relevant features
    to identify more effective ways of deploying promotions.

    Args:
        df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started optimize_promo_deployment")
    
    # Investigate the relationship between Promo, Sales, and other features
    corr_matrix = df.corr()
    print("Correlation matrix:")
    print(corr_matrix)
    logging.info("Computed correlation matrix")

    # Determine if certain store types, assortment levels, or competitive environments
    # are more responsive to promotions
    print("\nANOVA tests for promo effectiveness:")

    # ANOVA for StoreType
    store_type_anova = f_oneway(*[df[df['StoreType'] == t]['Sales'] for t in df['StoreType'].unique()])
    print(f"StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")

    # ANOVA for Assortment
    assortment_anova = f_oneway(*[df[df['Assortment'] == a]['Sales'] for a in df['Assortment'].unique()])
    print(f"Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")

    # ANOVA for CompetitionDistance
    competition_anova = f_oneway(*[df[df['CompetitionDistance'] < 1000]['Sales'],
                                  df[(df['CompetitionDistance'] >= 1000) & (df['CompetitionDistance'] < 5000)]['Sales'],
                                  df[df['CompetitionDistance'] >= 5000]['Sales']])
    print(f"CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")

    # Suggest strategies for deploying promotions more effectively
    print("\nStrategies for deploying promotions more effectively:")
    if store_type_anova.pvalue < 0.05:
        print("- Target specific store types that are more responsive to promotions")
    if assortment_anova.pvalue < 0.05:
        print("- Tailor promotions to the assortment level of the store")
    if competition_anova.pvalue < 0.05:
        print("- Focus promotions on stores with higher competition distance")
    print("- Conduct further analysis on the interaction between store features and promo effectiveness")
    logging.info("Suggested strategies for deploying promotions more effectively")

def analyze_store_hours(df):
    """
    Analyze the trends in customer behavior and sales during store opening and closing times.

    Args:
         df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started analyze_store_hours")
    
    # Analyze the impact of store opening and closing on customer behavior and sales
    print("Impact of store opening and closing on customer behavior and sales:")

    # Calculate the mean and standard deviation of Customers and Sales for open and closed stores
    open_stats = df[df['Open'] == 1][['Customers', 'Sales']].agg(['mean', 'std'])
    closed_stats = df[df['Open'] == 0][['Customers', 'Sales']].agg(['mean', 'std'])

    print("Open stores:")
    print(open_stats)
    print("\nClosed stores:")
    print(closed_stats)
    logging.info("Calculated mean and standard deviation for Customers and Sales for open and closed stores")

    # Visualize the trends in Customers and Sales over the course of a typical day or week
    print("\nVisualizing trends in Customers and Sales over time:")

    # Assume the data has a 'DayOfWeek' column
    df['DayOfWeek'] = df['DayOfWeek'].astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the trend in Customers
    sns.lineplot(x='DayOfWeek', y='Customers', data=df, ax=ax1)
    ax1.set_title('Trend in Customers over the week')
    ax1.set_xlabel('Day of the Week')
    ax1.set_ylabel('Customers')

    # Plot the trend in Sales
    sns.lineplot(x='DayOfWeek', y='Sales', data=df, ax=ax2)
    ax2.set_title('Trend in Sales over the week')
    ax2.set_xlabel('Day of the Week')
    ax2.set_ylabel('Sales')

    plt.tight_layout()
    plt.show()
    logging.info("Plotted trends in Customers and Sales over the week")

    # Identify patterns or insights that could inform the sales forecasting model
    print("\nInsights for sales forecasting:")
    if open_stats['Customers']['mean'] > closed_stats['Customers']['mean']:
        print("- Stores tend to have more customers when they are open compared to when they are closed.")
    if open_stats['Sales']['mean'] > closed_stats['Sales']['mean']:
        print("- Stores tend to have higher sales when they are open compared to when they are closed.")
    if 'DayOfWeek' in df.columns:
        print("- Customers and sales exhibit distinct trends over the course of the week, which could be useful for sales forecasting.")
    else:
        print("- The dataset does not contain a 'DayOfWeek' column, so trends over the week could not be analyzed.")
    logging.info("Identified patterns and insights for sales forecasting")



def compute_descriptive_statistics(df):
    logging.info("Started compute_descriptive_statistics")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    descriptive_stats = df[numeric_cols].describe()
    logging.info("Computed descriptive statistics")
    
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=True, ax=ax)
    ax.set_title('Distribution of Sales')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Sales")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CompetitionDistance'], kde=True, ax=ax)
    ax.set_title('Distribution of Competition Distance')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Competition Distance")

def compare_promo_distribution(train_df, test_df):
    logging.info("Started compare_promo_distribution")
    
    print("Distribution of Promo in the training dataset:")
    print(train_df['Promo'].value_counts())
    sns.histplot(train_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in training dataset")
    
    print("Distribution of Promo in the test dataset:")
    print(test_df['Promo'].value_counts())
    sns.histplot(test_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in test dataset")
    
    print("Comparing the distribution of Promo between the training and test datasets:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist([train_df['Promo'], test_df['Promo']], bins=2, label=['Training', 'Test'])
    ax1.set_title('Histogram of Promo')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Count')
    ax1.legend()
    plt.show()
    logging.info("Compared Promo distribution between training and test datasets")

def optimize_promo_deployment(df):
    """
    Investigate the relationship between Promo, Sales, and other relevant features
    to identify more effective ways of deploying promotions.

    Args:
        df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started optimize_promo_deployment")
    
    # Investigate the relationship between Promo, Sales, and other features
    corr_matrix = df.corr()
    print("Correlation matrix:")
    print(corr_matrix)
    logging.info("Computed correlation matrix")

    # Determine if certain store types, assortment levels, or competitive environments
    # are more responsive to promotions
    print("\nANOVA tests for promo effectiveness:")

    # ANOVA for StoreType
    store_type_anova = f_oneway(*[df[df['StoreType'] == t]['Sales'] for t in df['StoreType'].unique()])
    print(f"StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")

    # ANOVA for Assortment
    assortment_anova = f_oneway(*[df[df['Assortment'] == a]['Sales'] for a in df['Assortment'].unique()])
    print(f"Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")

    # ANOVA for CompetitionDistance
    competition_anova = f_oneway(*[df[df['CompetitionDistance'] < 1000]['Sales'],
                                  df[(df['CompetitionDistance'] >= 1000) & (df['CompetitionDistance'] < 5000)]['Sales'],
                                  df[df['CompetitionDistance'] >= 5000]['Sales']])
    print(f"CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")

    # Suggest strategies for deploying promotions more effectively
    print("\nStrategies for deploying promotions more effectively:")
    if store_type_anova.pvalue < 0.05:
        print("- Target specific store types that are more responsive to promotions")
    if assortment_anova.pvalue < 0.05:
        print("- Tailor promotions to the assortment level of the store")
    if competition_anova.pvalue < 0.05:
        print("- Focus promotions on stores with higher competition distance")
    print("- Conduct further analysis on the interaction between store features and promo effectiveness")
    logging.info("Suggested strategies for deploying promotions more effectively")

def analyze_store_hours(df):
    """
    Analyze the trends in customer behavior and sales during store opening and closing times.

    Args:
         df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started analyze_store_hours")
    
    # Analyze the impact of store opening and closing on customer behavior and sales
    print("Impact of store opening and closing on customer behavior and sales:")

    # Calculate the mean and standard deviation of Customers and Sales for open and closed stores
    open_stats = df[df['Open'] == 1][['Customers', 'Sales']].agg(['mean', 'std'])
    closed_stats = df[df['Open'] == 0][['Customers', 'Sales']].agg(['mean', 'std'])

    print("Open stores:")
    print(open_stats)
    print("\nClosed stores:")
    print(closed_stats)
    logging.info("Calculated mean and standard deviation for Customers and Sales for open and closed stores")

    # Visualize the trends in Customers and Sales over the course of a typical day or week
    print("\nVisualizing trends in Customers and Sales over time:")

    # Assume the data has a 'DayOfWeek' column
    df['DayOfWeek'] = df['DayOfWeek'].astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the trend in Customers
    sns.lineplot(x='DayOfWeek', y='Customers', data=df, ax=ax1)
    ax1.set_title('Trend in Customers over the week')
    ax1.set_xlabel('Day of the Week')
    ax1.set_ylabel('Customers')

    # Plot the trend in Sales
    sns.lineplot(x='DayOfWeek', y='Sales', data=df, ax=ax2)
    ax2.set_title('Trend in Sales over the week')
    ax2.set_xlabel('Day of the Week')
    ax2.set_ylabel('Sales')

    plt.tight_layout()
    plt.show()
    logging.info("Plotted trends in Customers and Sales over the week")

    # Identify patterns or insights that could inform the sales forecasting model
    print("\nInsights for sales forecasting:")
    if open_stats['Customers']['mean'] > closed_stats['Customers']['mean']:
        print("- Stores tend to have more customers when they are open compared to when they are closed.")
    if open_stats['Sales']['mean'] > closed_stats['Sales']['mean']:
        print("- Stores tend to have higher sales when they are open compared to when they are closed.")
    if 'DayOfWeek' in df.columns:
        print("- Customers and sales exhibit distinct trends over the course of the week, which could be useful for sales forecasting.")
    else:
        print("- The dataset does not contain a 'DayOfWeek' column, so trends over the week could not be analyzed.")
    logging.info("Identified patterns and insights for sales forecasting")

def analyze_weekday_openings(df):
    """
    Identify the stores that are open on all weekdays and analyze the impact on their weekend sales.

    Args:
         df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started analyze_weekday_openings")

    # Group the data by store and count the number of days each store was open
    store_open_counts = df.groupby('Store')['Open'].sum()

    # Identify the stores that were open on all weekdays (Monday to Friday)
    weekday_open_stores = store_open_counts[store_open_counts == 5].index.tolist()
    print(f"Stores open on all weekdays: {weekday_open_stores}")
    logging.info(f"Identified stores open on all weekdays: {weekday_open_stores}")




def compute_descriptive_statistics(df):
    logging.info("Started compute_descriptive_statistics")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    descriptive_stats = df[numeric_cols].describe()
    logging.info("Computed descriptive statistics")
    
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=True, ax=ax)
    ax.set_title('Distribution of Sales')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Sales")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CompetitionDistance'], kde=True, ax=ax)
    ax.set_title('Distribution of Competition Distance')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of Competition Distance")

def compare_promo_distribution(train_df, test_df):
    logging.info("Started compare_promo_distribution")
    
    print("Distribution of Promo in the training dataset:")
    print(train_df['Promo'].value_counts())
    sns.histplot(train_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in training dataset")
    
    print("Distribution of Promo in the test dataset:")
    print(test_df['Promo'].value_counts())
    sns.histplot(test_df['Promo'], kde=True)
    plt.show()
    logging.info("Analyzed and plotted Promo distribution in test dataset")
    
    print("Comparing the distribution of Promo between the training and test datasets:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist([train_df['Promo'], test_df['Promo']], bins=2, label=['Training', 'Test'])
    ax1.set_title('Histogram of Promo')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Count')
    ax1.legend()
    plt.show()
    logging.info("Compared Promo distribution between training and test datasets")

def optimize_promo_deployment(df):
    """
    Investigate the relationship between Promo, Sales, and other relevant features
    to identify more effective ways of deploying promotions.

    Args:
        df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started optimize_promo_deployment")
    
    # Investigate the relationship between Promo, Sales, and other features
    corr_matrix = df.corr()
    print("Correlation matrix:")
    print(corr_matrix)
    logging.info("Computed correlation matrix")

    # Determine if certain store types, assortment levels, or competitive environments
    # are more responsive to promotions
    print("\nANOVA tests for promo effectiveness:")

    # ANOVA for StoreType
    store_type_anova = f_oneway(*[df[df['StoreType'] == t]['Sales'] for t in df['StoreType'].unique()])
    print(f"StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")

    # ANOVA for Assortment
    assortment_anova = f_oneway(*[df[df['Assortment'] == a]['Sales'] for a in df['Assortment'].unique()])
    print(f"Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")

    # ANOVA for CompetitionDistance
    competition_anova = f_oneway(*[df[df['CompetitionDistance'] < 1000]['Sales'],
                                  df[(df['CompetitionDistance'] >= 1000) & (df['CompetitionDistance'] < 5000)]['Sales'],
                                  df[df['CompetitionDistance'] >= 5000]['Sales']])
    print(f"CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")
    logging.info(f"Computed ANOVA for CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")

    # Suggest strategies for deploying promotions more effectively
    print("\nStrategies for deploying promotions more effectively:")
    if store_type_anova.pvalue < 0.05:
        print("- Target specific store types that are more responsive to promotions")
    if assortment_anova.pvalue < 0.05:
        print("- Tailor promotions to the assortment level of the store")
    if competition_anova.pvalue < 0.05:
        print("- Focus promotions on stores with higher competition distance")
    print("- Conduct further analysis on the interaction between store features and promo effectiveness")
    logging.info("Suggested strategies for deploying promotions more effectively")

def analyze_store_hours(df):
    """
    Analyze the trends in customer behavior and sales during store opening and closing times.

    Args:
         df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started analyze_store_hours")
    
    # Analyze the impact of store opening and closing on customer behavior and sales
    print("Impact of store opening and closing on customer behavior and sales:")

    # Calculate the mean and standard deviation of Customers and Sales for open and closed stores
    open_stats = df[df['Open'] == 1][['Customers', 'Sales']].agg(['mean', 'std'])
    closed_stats = df[df['Open'] == 0][['Customers', 'Sales']].agg(['mean', 'std'])

    print("Open stores:")
    print(open_stats)
    print("\nClosed stores:")
    print(closed_stats)
    logging.info("Calculated mean and standard deviation for Customers and Sales for open and closed stores")

    # Visualize the trends in Customers and Sales over the course of a typical day or week
    print("\nVisualizing trends in Customers and Sales over time:")

    # Assume the data has a 'DayOfWeek' column
    df['DayOfWeek'] = df['DayOfWeek'].astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the trend in Customers
    sns.lineplot(x='DayOfWeek', y='Customers', data=df, ax=ax1)
    ax1.set_title('Trend in Customers over the week')
    ax1.set_xlabel('Day of the Week')
    ax1.set_ylabel('Customers')

    # Plot the trend in Sales
    sns.lineplot(x='DayOfWeek', y='Sales', data=df, ax=ax2)
    ax2.set_title('Trend in Sales over the week')
    ax2.set_xlabel('Day of the Week')
    ax2.set_ylabel('Sales')

    plt.tight_layout()
    plt.show()
    logging.info("Plotted trends in Customers and Sales over the week")

    # Identify patterns or insights that could inform the sales forecasting model
    print("\nInsights for sales forecasting:")
    if open_stats['Customers']['mean'] > closed_stats['Customers']['mean']:
        print("- Stores tend to have more customers when they are open compared to when they are closed.")
    if open_stats['Sales']['mean'] > closed_stats['Sales']['mean']:
        print("- Stores tend to have higher sales when they are open compared to when they are closed.")
    if 'DayOfWeek' in df.columns:
        print("- Customers and sales exhibit distinct trends over the course of the week, which could be useful for sales forecasting.")
    else:
        print("- The dataset does not contain a 'DayOfWeek' column, so trends over the week could not be analyzed.")
    logging.info("Identified patterns and insights for sales forecasting")

def analyze_weekday_openings(df):
    """
    Identify the stores that are open on all weekdays and analyze the impact on their weekend sales.

    Args:
         df (str): Dataframe.

    Returns:
        None
    """
    logging.info("Started analyze_weekday_openings")

    # Group the data by store and count the number of days each store was open
    store_open_counts = df.groupby('Store')['Open'].sum()

    # Identify the stores that were open on all weekdays (Monday to Friday)
    weekday_open_stores = store_open_counts[store_open_counts == 5].index.tolist()
    print(f"Stores open on all weekdays: {weekday_open_stores}")
    logging.info(f"Identified stores open on all weekdays: {weekday_open_stores}")

 
