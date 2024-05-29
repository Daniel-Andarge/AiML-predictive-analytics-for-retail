import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose


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