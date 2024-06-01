import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def explore_target_variable(data, target_variable):
    """
    Explores the distribution of the target variable.

    Parameters:
    data (pandas.DataFrame): The input dataset.
    target_variable (str): The name of the target variable.

    Returns:
    None
    """
    print(f'Target variable: {target_variable}')
    print(data[target_variable].value_counts())

    plt.figure(figsize=(10, 6))
    data[target_variable].plot(kind='hist')
    plt.title('Distribution of Target Variable')
    plt.xlabel(target_variable)
    plt.ylabel('Count')
    plt.show()

def analyze_feature_target_relationship(data, target_variable):
    """
    Analyzes the relationship between features and the target variable.

    Parameters:
    data (pandas.DataFrame): The input dataset.
    target_variable (str): The name of the target variable.

    Returns:
    None
    """
    # Continuous features
    continuous_features = [col for col in data.columns if data[col].dtype != 'object']
    for feature in continuous_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature, y=target_variable, data=data)
        plt.title(f'Relationship between {feature} and {target_variable}')
        plt.xlabel(feature)
        plt.ylabel(target_variable)
        plt.show()

    # Categorical features
    categorical_features = [col for col in data.columns if data[col].dtype == 'object']
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        data.groupby(feature)[target_variable].mean().plot(kind='bar')
        plt.title(f'Relationship between {feature} and {target_variable}')
        plt.xlabel(feature)
        plt.ylabel(target_variable)
        plt.show()

def check_multicollinearity(data):
    """
    Checks for multicollinearity in the dataset.

    Parameters:
    data (pandas.DataFrame): The input dataset.

    Returns:
    None
    """
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
    plt.title('Correlation Matrix')
    plt.show()
    return corr_matrix


def test_nonlinear_correlation(df):
    """
    Test for nonlinear correlation between two variables.
    
    Parameters:
    X (numpy array): Independent variable (e.g., Promo2)
    y (numpy array): Dependent variable (e.g., Sales)
    
    Returns:
    float: Spearman's rank correlation coefficient
    """

    X=df['Sales']
    y=df['Promo2']
    # Calculate Pearson's correlation coefficient
    pearson_corr = np.corrcoef(X, y)[0, 1]
    
    # Calculate Spearman's rank correlation coefficient
    spearman_corr, _ = spearmanr(X, y)
    
    # Print the results
    print(f"Pearson's correlation coefficient: {pearson_corr:.3f}")
    print(f"Spearman's rank correlation coefficient: {spearman_corr:.3f}")
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y)
    plt.xlabel("Promo2")
    plt.ylabel("Sales")
    plt.title("Scatter Plot of Promo2 vs. Sales")
    plt.show()
    
    return spearman_corr

# 8. Perform feature engineering (if necessary)
# Add, remove, or transform features as per your requirements

# 9. Split the data into training and testing sets
""" from sklearn.model_selection import train_test_split
X = data.drop(target_variable, axis=1)
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) """

# 10. Scale the features (if necessary)
""" from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 """
# 11. Train your model
# Import and train your preferred machine learning model here