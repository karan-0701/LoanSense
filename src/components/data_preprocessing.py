import pandas as pd
import numpy as np

def load_data(file_path):
    """Load CSV data into a DataFrame."""
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def drop_empty_columns(df):
    """Drop columns that are completely empty."""
    empty_columns = df.columns[df.isna().all()]
    df.drop(empty_columns, axis=1, inplace=True)
    # print(f"Dropped empty columns: {list(empty_columns)}")
    return df

def clean_loan_status(df):
    """Simplify loan_status to Fully Paid or Default."""
    df['loan_status'] = df['loan_status'].replace({'Charged Off': 'Default'})
    df = df[df['loan_status'].isin(['Fully Paid', 'Default'])]
    print("Loan status distribution:")
    print(df['loan_status'].value_counts())
    return df

def drop_high_missing(df, threshold=60):
    """Drop columns with more than threshold% missing values."""
    percent_missing = df.isnull().sum() * 100 / len(df)
    cols_to_drop = percent_missing[percent_missing >= threshold].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    # print(f"Dropped columns with >={threshold}% missing: {list(cols_to_drop)}")
    return df

def drop_irrelevant_columns(df):
    """Drop ID and URL or other irrelevant/redundant columns."""
    cols_to_drop = ['id', 'url', 'policy_code', 'title', 'zip_code', 'pymnt_plan',
                    'funded_amnt', 'funded_amnt_inv', 'out_prncp_inv', 'hardship_flag', 'emp_title', 'addr_state']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    # print(f"Dropped irrelevant/redundant columns: {cols_to_drop}")
    return df

def drop_highly_correlated(df, threshold=0.9):
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()

    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for correlated_col in high_corr:
            if correlated_col not in to_drop and col not in to_drop:
                if numeric_df[col].var() >= numeric_df[correlated_col].var():
                    to_drop.add(correlated_col)
                else:
                    to_drop.add(col)

    df.drop(columns=list(to_drop), inplace=True)
    # print("Columns dropped due to high correlation (kept highest variance):", list(to_drop))
    return df


def process_dates(df):
    # Process sub_grade
    if 'sub_grade' in df.columns:
        df['sub_grade'] = df['sub_grade'].str[1:].fillna(-1).astype(int)
    
    # Date columns to convert
    date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df.drop(columns=[col], inplace=True)
    return df


def fill_missing_values(df):
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical/string columns with 'missing'
    categorical_cols = ['disbursement_method', 'debt_settlement_flag',
                        'application_type', 'initial_list_status',
                        'purpose', 'loan_status', 'verification_status',
                        'home_ownership', 'emp_length', 'grade', 'term']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
    
    return df

def preprocess(file_path):
    df = load_data(file_path)
    df = drop_empty_columns(df)
    df = clean_loan_status(df)
    df = drop_high_missing(df, threshold=60)
    
    df = drop_irrelevant_columns(df)
    df = drop_highly_correlated(df, threshold=0.9)
    df = process_dates(df)
    df = fill_missing_values(df)
    
    print("Preprocessing complete. Final shape:", df.shape)
    return df