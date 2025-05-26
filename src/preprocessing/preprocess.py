import pandas as pd

def preprocess():
    df = pd.read_csv('data/processed/clean_raw.csv')

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    df = df.dropna(subset=['MSRP'])
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df.to_csv('data/processed/train.csv', index=False)
    print("✅ Preprocessing completed.")

if __name__ == '__main__':
    preprocess()
