import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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



def explore_promo_distribution(train_df, test_df):
    """
    Explore the distribution of promotions between the training and test sets.
    
    Parameters:
    train_df (pandas.DataFrame): The training set DataFrame.
    test_df (pandas.DataFrame): The test set DataFrame.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the histogram for the training set
    train_df['Promo'].hist(ax=ax1, bins=10, edgecolor='black', color='lightblue')
    ax1.set_title('Promo Distribution - Training Set')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Count')
    
    # Plot the histogram for the test set
    test_df['Promo'].hist(ax=ax2, bins=10, edgecolor='black', color='lightgreen')
    ax2.set_title('Promo Distribution - Test Set')
    ax2.set_xlabel('Promo')
    ax2.set_ylabel('Count')
    
    # Add a legend
    ax1.legend(['Training Set'])
    ax2.legend(['Test Set'])
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Display the plot
    plt.show()



def analyze_sales_around_holidays(df):
    """
    Analyze sales behavior around holidays.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Group the data by StateHoliday and SchoolHoliday
    holiday_sales = df.groupby(['StateHoliday', 'SchoolHoliday'])['Sales'].mean().reset_index()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the line plot for StateHoliday
    sns.lineplot(x='StateHoliday', y='Sales', data=holiday_sales, ax=ax1)
    ax1.set_title('Sales by State Holiday')
    ax1.set_xlabel('State Holiday')
    ax1.set_ylabel('Sales')
    ax1.set_xticklabels(['No Holiday', 'Public Holiday', 'Easter', 'Christmas'])
    
    # Plot the line plot for SchoolHoliday
    sns.lineplot(x='SchoolHoliday', y='Sales', data=holiday_sales, ax=ax2)
    ax2.set_title('Sales by School Holiday')
    ax2.set_xlabel('School Holiday')
    ax2.set_ylabel('Sales')
    ax2.set_xticklabels(['No Holiday', 'Holiday'])
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Display the plot
    plt.show()



def investigate_seasonal_behaviors(df):
    """
    Investigate seasonal purchase behaviors in the data.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Convert the date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group the data by month and calculate the mean sales
    monthly_sales = df.groupby(df['Date'].dt.month)['Sales'].mean().reset_index()
    monthly_sales.columns = ['Month', 'Sales']
    
    # Group the data by quarter and calculate the mean sales
    quarterly_sales = df.groupby(df['Date'].dt.quarter)['Sales'].mean().reset_index()
    quarterly_sales.columns = ['Quarter', 'Sales']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the monthly sales
    sns.barplot(x='Month', y='Sales', data=monthly_sales, ax=ax1)
    ax1.set_title('Monthly Sales')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Sales')
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Plot the quarterly sales
    sns.barplot(x='Quarter', y='Sales', data=quarterly_sales, ax=ax2)
    ax2.set_title('Quarterly Sales')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Sales')
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Display the plot
    plt.show()



def examine_sales_customers_relationship(df):
    """
    Examine the relationship between sales and customers.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Create a scatter plot of sales vs. customers
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Customers', y='Sales', data=df, ax=ax)
    ax.set_title('Relationship Between Sales and Customers')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Sales')
    
    # Calculate the correlation coefficient
    correlation = df['Customers'].corr(df['Sales'])
    
    # Add the correlation coefficient to the plot
    ax.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax.transAxes, va='top')
    
    # Display the plot
    plt.show()




def analyze_promotion_impact(df):
    """
    Analyze the impact of promotions on sales and customer behavior.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Group the data by Promo and Promo2
    promo_data = df.groupby(['Promo', 'Promo2'])['Sales', 'Customers'].mean().reset_index()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the impact of Promo on sales and customers
    sns.barplot(x='Promo', y='Sales', data=promo_data, ax=ax1)
    ax1.set_title('Impact of Promo on Sales')
    ax1.set_xlabel('Promo')
    ax1.set_ylabel('Sales')
    
    sns.barplot(x='Promo', y='Customers', data=promo_data, ax=ax2)
    ax2.set_title('Impact of Promo on Customers')
    ax2.set_xlabel('Promo')
    ax2.set_ylabel('Customers')
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Display the plot
    plt.show()
    
    # Group the data by Promo2 and calculate the mean sales and customers
    promo2_data = df.groupby('Promo2')['Sales', 'Customers'].mean().reset_index()
    
    # Create a line plot to show the impact of Promo2 on sales and customers
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='Promo2', y='Sales', data=promo2_data, label='Sales', ax=ax)
    sns.lineplot(x='Promo2', y='Customers', data=promo2_data, label='Customers', ax=ax)
    ax.set_title('Impact of Promo2 on Sales and Customers')
    ax.set_xlabel('Promo2')
    ax.set_ylabel('Sales and Customers')
    ax.legend()
    
    # Display the plot
    plt.show()




def explore_store_opening_trends(df):
    """
    Explore the store opening and closing trends and their impact on sales.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Group the data by store and day of the week
    store_weekday_data = df.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()
    
    # Create a pivot table to visualize the sales patterns
    store_weekday_pivot = store_weekday_data.pivot(index='Store', columns='DayOfWeek', values='Sales')
    
    # Create a heatmap to visualize the sales patterns
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(store_weekday_pivot, cmap='YlOrRd', annot=True, ax=ax)
    ax.set_title('Sales Patterns by Store and Weekday')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Store')
    plt.show()
    
    # Analyze the sales on weekends
    weekend_data = df[df['DayOfWeek'].isin([6, 7])]
    
    # Group the weekend data by store and opening/closing times
    weekend_data['OpeningTime'] = df['Open'].apply(lambda x: 'Open' if x == 1 else 'Closed')
    weekend_data['ClosingTime'] = df['Close'].apply(lambda x: 'Open' if x == 1 else 'Closed')
    
    weekend_opening_data = weekend_data.groupby(['Store', 'OpeningTime'])['Sales'].mean().reset_index()
    weekend_closing_data = weekend_data.groupby(['Store', 'ClosingTime'])['Sales'].mean().reset_index()
    
    # Create bar plots to visualize the impact of opening and closing times on weekend sales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(x='OpeningTime', y='Sales', data=weekend_opening_data, ax=ax1)
    ax1.set_title('Impact of Opening Time on Weekend Sales')
    ax1.set_xlabel('Opening Time')
    ax1.set_ylabel('Sales')
    
    sns.barplot(x='ClosingTime', y='Sales', data=weekend_closing_data, ax=ax2)
    ax2.set_title('Impact of Closing Time on Weekend Sales')
    ax2.set_xlabel('Closing Time')
    ax2.set_ylabel('Sales')
    
    plt.subplots_adjust(wspace=0.4)
    plt.show()



def assess_assortment_impact(df):
    """
    Assess the impact of the Assortment feature on sales.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Group the data by Assortment and calculate the mean sales
    assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()
    
    # Create a bar plot to visualize the sales by Assortment
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Assortment', y='Sales', data=assortment_sales, ax=ax)
    ax.set_title('Sales by Assortment')
    ax.set_xlabel('Assortment')
    ax.set_ylabel('Average Sales')
    
    # Add labels to the bars
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.2f}", ha='center', va='bottom')
    
    plt.show()
    
    # Analyze the distribution of sales for each Assortment type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Assortment', y='Sales', data=df, ax=ax)
    ax.set_title('Distribution of Sales by Assortment')
    ax.set_xlabel('Assortment')
    ax.set_ylabel('Sales')
    
    plt.show()



def investigate_competition_impact(df):
    """
    Investigate the impact of competition on sales.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    """
    # Analyze the relationship between CompetitionDistance and sales
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df, ax=ax)
    ax.set_title('Relationship between Competition Distance and Sales')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Sales')
    plt.show()
    
    # Investigate the impact of competition distance based on store location
    fig, ax = plt.subplots(figsize=(10, 6))
    df['StoreType'].replace({'a': 'City Center', 'b': 'Suburban', 'c': 'Regional Center', 'd': 'Rural'}, inplace=True)
    sns.scatterplot(x='CompetitionDistance', y='Sales', hue='StoreType', data=df, ax=ax)
    ax.set_title('Relationship between Competition Distance and Sales by Store Type')
    ax.set_xlabel('Competition Distance')
    ax.set_ylabel('Sales')
    plt.legend()
    plt.show()
    
    # Analyze the impact of new competitors on existing store sales
    new_competitors = df[df['CompetitionOpenSinceMonth'].notnull() & df['CompetitionOpenSinceYear'].notnull()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Date', y='Sales', data=new_competitors, hue='Store')
    ax.set_title('Impact of New Competitors on Existing Store Sales')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    plt.legend(title='Store')
    plt.show()