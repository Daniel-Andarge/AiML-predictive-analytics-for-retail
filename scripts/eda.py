import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr, f_oneway

def compute_descriptive_statistics(df):
    """
    Compute descriptive statistics for the input DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Compute descriptive statistics for the numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    descriptive_stats = df[numeric_cols].describe()
    
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    # Analyze the distribution of the 'Sales' variable
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=True, ax=ax)
    ax.set_title('Distribution of Sales')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    plt.show()
    
    # Analyze the distribution of the 'CompetitionDistance' variable
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CompetitionDistance'], kde=True, ax=ax)
    ax.set_title('Distribution of Competition Distance')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Frequency')
    plt.show()



def compare_promo_distribution(train_df, test_df):
    """
    Compare the distribution of the 'Promo' feature between the training and test datasets.

    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.

    Returns:
        None
    """
    # Inspect the distribution of Promo in the training dataset
    print("Distribution of Promo in the training dataset:")
    print(train_df['Promo'].value_counts())
    sns.histplot(train_df['Promo'], kde=True)
    plt.show()

    # Inspect the distribution of Promo in the test dataset
    print("Distribution of Promo in the test dataset:")
    print(test_df['Promo'].value_counts())
    sns.histplot(test_df['Promo'], kde=True)
    plt.show()

    # Compare the distribution of Promo between the two datasets
    print("Comparing the distribution of Promo between the training and test datasets:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist([train_df['Promo'], test_df['Promo']], bins=2, label=['Training', 'Test'])
    ax1.set_title('Histogram of Promo')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Count')
    ax1.legend()

    # Box plot
    ax2.boxplot([train_df['Promo'], test_df['Promo']], labels=['Training', 'Test'])
    ax2.set_title('Box Plot of Promo')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Promo')

    plt.show()

    # Check if the distribution of Promo is similar between the two datasets
    train_promo_count = train_df['Promo'].value_counts()
    test_promo_count = test_df['Promo'].value_counts()

    if train_promo_count.equals(test_promo_count):
        print("The distribution of Promo is similar between the training and test datasets.")
    else:
        print("The distribution of Promo is not similar between the training and test datasets.")





def analyze_sales_around_holidays(data_path):
    """
    Analyze the sales behavior before, during, and after holidays.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Group the data by StateHoliday and calculate summary statistics
    holiday_stats = df.groupby('StateHoliday')['Sales'].agg(['mean', 'median', 'std'])
    print("Sales statistics by holiday type:")
    print(holiday_stats)

    # Visualize the sales behavior around holidays
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Line plot of sales over time, colored by holiday type
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days_Since_Holiday'] = (df['Date'] - df['Date'].where(df['StateHoliday'] != '0').ffill()).dt.days
    sns.lineplot(x='Days_Since_Holiday', y='Sales', hue='StateHoliday', data=df, ax=ax1)
    ax1.set_title('Sales Around Holidays')
    ax1.set_xlabel('Days Since Holiday')
    ax1.set_ylabel('Sales')
    ax1.legend(title='Holiday Type')

    # Box plot of sales by holiday type
    sns.boxplot(x='StateHoliday', y='Sales', data=df, ax=ax2)
    ax2.set_title('Sales by Holiday Type')
    ax2.set_xlabel('Holiday Type')
    ax2.set_ylabel('Sales')

    plt.show()

    # Check for any patterns or trends in sales around holidays
    print("Observations:")
    if df.loc[df['StateHoliday'] != '0', 'Sales'].mean() < df.loc[df['StateHoliday'] == '0', 'Sales'].mean():
        print("Sales tend to be lower during state holidays.")
    else:
        print("Sales tend to be higher during state holidays.")

    if df['Days_Since_Holiday'].max() > 0:
        print("Sales appear to recover after state holidays.")
    else:
        print("No clear pattern in sales after state holidays.")




def analyze_seasonal_purchase_behavior(data_path):
    """
    Analyze seasonal purchase behaviors in the data.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Investigate seasonal patterns in sales
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Seasonal decomposition of sales
    result = seasonal_decompose(df['Sales'], model='additive', period=365)
    result.plot(ax=ax1)
    ax1.set_title('Seasonal Decomposition of Sales')

    # Seasonal plot of sales
    sns.lineplot(x='Date', y='Sales', data=df, ax=ax2)
    ax2.set_title('Seasonal Plot of Sales')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales')

    plt.show()

    # Explore the impact of Promo2 on seasonal sales patterns
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Calculate average sales by month and Promo2 status
    promo2_sales = df.groupby(['Month', 'Promo2'])['Sales'].mean().unstack()

    # Plot the impact of Promo2 on seasonal sales
    promo2_sales.plot(title='Impact of Promo2 on Seasonal Sales Patterns')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.show()

    # Check for observations
    print("Observations:")
    if promo2_sales['0'].max() > promo2_sales['1'].max():
        print("Sales appear to be higher during periods without Promo2.")
    else:
        print("Sales appear to be higher during periods with Promo2.")

    if result.seasonal.max() > result.seasonal.min():
        print("There are clear seasonal patterns in sales.")
    else:
        print("No clear seasonal patterns in sales.")



def analyze_sales_customers_correlation(data_path):
    """
    Analyze the correlation between sales and the number of customers.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Calculate the correlation coefficient
    correlation = df['Sales'].corr(df['Customers'])
    print(f"Correlation coefficient between Sales and Customers: {correlation:.2f}")

    # Visualize the relationship
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Customers', y='Sales', data=df, ax=ax)
    ax.set_title('Relationship between Sales and Customers')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Sales')

    # Add a regression line
    slope, intercept = np.polyfit(df['Customers'], df['Sales'], 1)
    ax.plot(df['Customers'], slope * df['Customers'] + intercept, color='r', label=f'Regression line (slope={slope:.2f})')
    ax.legend()

    plt.show()

    # Discuss the implications for the sales forecasting model
    print("Implications for the sales forecasting model:")
    if correlation > 0:
        print("The positive correlation between sales and the number of customers suggests that the sales forecasting model should incorporate the customer count as a feature. This could improve the model's ability to predict sales accurately.")
    elif correlation < 0:
        print("The negative correlation between sales and the number of customers suggests that the sales forecasting model should consider the customer count as a feature, but with an inverse relationship. This could help the model better capture the dynamics between sales and customer count.")
    else:
        print("The lack of correlation between sales and the number of customers suggests that the customer count may not be a useful feature for the sales forecasting model. The model should focus on other factors that are more strongly related to sales.")





def analyze_promo_impact(data_path):
    """
    Analyze the impact of promotions on sales and customers.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Group the data by Promo and calculate summary statistics
    promo_stats = df.groupby('Promo')[['Sales', 'Customers']].agg(['mean', 'std', 'count'])
    print("Summary statistics by Promo:")
    print(promo_stats)

    # Perform statistical tests
    # t-test to compare sales between promo and non-promo groups
    t_stat, p_value = ttest_ind(df[df['Promo'] == 1]['Sales'], df[df['Promo'] == 0]['Sales'])
    print(f"\nT-test for Sales: t-statistic={t_stat:.2f}, p-value={p_value:.4f}")

    # ANOVA to compare sales and customers between promo groups
    sales_anova = f_oneway(df[df['Promo'] == 1]['Sales'], df[df['Promo'] == 0]['Sales'])
    customers_anova = f_oneway(df[df['Promo'] == 1]['Customers'], df[df['Promo'] == 0]['Customers'])
    print(f"ANOVA for Sales: F-statistic={sales_anova.statistic:.2f}, p-value={sales_anova.pvalue:.4f}")
    print(f"ANOVA for Customers: F-statistic={customers_anova.statistic:.2f}, p-value={customers_anova.pvalue:.4f}")

    # Analyze the impact on new and existing customers
    new_customers = df[df['Promo'] == 1]['Customers'] - df[df['Promo'] == 0]['Customers']
    existing_customers = df[df['Promo'] == 0]['Customers']
    print("\nImpact on new and existing customers:")
    print(f"Average increase in new customers during promotions: {new_customers.mean():.2f}")
    print(f"Average number of existing customers without promotions: {existing_customers.mean():.2f}")




def optimize_promo_deployment(data_path):
    """
    Investigate the relationship between Promo, Sales, and other relevant features
    to identify more effective ways of deploying promotions.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Investigate the relationship between Promo, Sales, and other features
    corr_matrix = df.corr()
    print("Correlation matrix:")
    print(corr_matrix)

    # Determine if certain store types, assortment levels, or competitive environments
    # are more responsive to promotions
    print("\nANOVA tests for promo effectiveness:")

    # ANOVA for StoreType
    store_type_anova = f_oneway(*[df[df['StoreType'] == t]['Sales'] for t in df['StoreType'].unique()])
    print(f"StoreType: F-statistic={store_type_anova.statistic:.2f}, p-value={store_type_anova.pvalue:.4f}")

    # ANOVA for Assortment
    assortment_anova = f_oneway(*[df[df['Assortment'] == a]['Sales'] for a in df['Assortment'].unique()])
    print(f"Assortment: F-statistic={assortment_anova.statistic:.2f}, p-value={assortment_anova.pvalue:.4f}")

    # ANOVA for CompetitionDistance
    competition_anova = f_oneway(*[df[df['CompetitionDistance'] < 1000]['Sales'],
                                  df[(df['CompetitionDistance'] >= 1000) & (df['CompetitionDistance'] < 5000)]['Sales'],
                                  df[df['CompetitionDistance'] >= 5000]['Sales']])
    print(f"CompetitionDistance: F-statistic={competition_anova.statistic:.2f}, p-value={competition_anova.pvalue:.4f}")

    # Suggest strategies for deploying promotions more effectively
    print("\nStrategies for deploying promotions more effectively:")
    if store_type_anova.pvalue < 0.05:
        print("- Target specific store types that are more responsive to promotions")
    if assortment_anova.pvalue < 0.05:
        print("- Tailor promotions to the assortment level of the store")
    if competition_anova.pvalue < 0.05:
        print("- Focus promotions on stores with higher competition distance")
    print("- Conduct further analysis on the interaction between store features and promo effectiveness")




def analyze_store_hours(data_path):
    """
    Analyze the trends in customer behavior and sales during store opening and closing times.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Analyze the impact of store opening and closing on customer behavior and sales
    print("Impact of store opening and closing on customer behavior and sales:")

    # Calculate the mean and standard deviation of Customers and Sales for open and closed stores
    open_stats = df[df['Open'] == 1][['Customers', 'Sales']].agg(['mean', 'std'])
    closed_stats = df[df['Open'] == 0][['Customers', 'Sales']].agg(['mean', 'std'])

    print("Open stores:")
    print(open_stats)
    print("\nClosed stores:")
    print(closed_stats)

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


def analyze_weekday_openings(data_path):
    """
    Identify the stores that are open on all weekdays and analyze the impact on their weekend sales.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Group the data by store and count the number of days each store was open
    store_open_counts = df.groupby('Store')['Open'].sum()

    # Identify the stores that were open on all weekdays (Monday to Friday)
    weekday_open_stores = store_open_counts[store_open_counts == 5].index.tolist()
    print(f"Stores open on all weekdays: {weekday_open_stores}")

    # Calculate the average weekend sales for weekday-open stores and other stores
    weekend_sales = df[df['DayOfWeek'].isin([6, 7])][['Store', 'Sales']]
    weekday_open_sales = weekend_sales[weekend_sales['Store'].isin(weekday_open_stores)]['Sales']
    other_sales = weekend_sales[~weekend_sales['Store'].isin(weekday_open_stores)]['Sales']

    weekday_open_mean = weekday_open_sales.mean()
    other_mean = other_sales.mean()

    print(f"Average weekend sales for stores open on all weekdays: {weekday_open_mean:.2f}")
    print(f"Average weekend sales for other stores: {other_mean:.2f}")

    # Visualize the comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Stores open on all weekdays', 'Other stores'], [weekday_open_mean, other_mean])
    ax.set_title('Average Weekend Sales')
    ax.set_xlabel('Store Type')
    ax.set_ylabel('Sales')
    plt.show()