import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class RetailAnalyzer:
    def __init__(self):
        self.df = None

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read file: {e}")

        expected = {"Date","Product","Category","Price","Quantity Sold","Total Sales"}
        if not expected.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns. Expected at least: {expected}")

        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')

        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce')
        df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')

        recompute_mask = df['Total Sales'].isna() & df['Price'].notna() & df['Quantity Sold'].notna()
        df.loc[recompute_mask, 'Total Sales'] = (df.loc[recompute_mask, 'Price'] * df.loc[recompute_mask, 'Quantity Sold']).round(2)

        self.df = df
        return df

    def validate(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        issues = {}
        issues['missing_product'] = int(self.df['Product'].isna().sum())
        issues['missing_price'] = int(self.df['Price'].isna().sum())
        issues['missing_quantity'] = int(self.df['Quantity Sold'].isna().sum())
        issues['invalid_price_negative'] = int((self.df['Price'] < 0).sum())
        issues['bad_dates'] = int(self.df['Date_parsed'].isna().sum())

        return issues

    def calculate_metrics(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        df = self.df.copy()
        total_sales = float(df['Total Sales'].sum(skipna=True))
        avg_sales = float(df['Total Sales'].mean(skipna=True))

        pop = df.dropna(subset=['Product','Quantity Sold']).groupby('Product')['Quantity Sold'].sum()
        if not pop.empty:
            most_popular_product = pop.idxmax()
            most_popular_qty = int(pop.max())
        else:
            most_popular_product = None
            most_popular_qty = 0

        metrics = {
            'total_sales': round(total_sales,2),
            'average_sales_per_row': round(avg_sales,2),
            'most_popular_product': most_popular_product,
            'most_popular_qty': most_popular_qty
        }
        return metrics

    def filter_data(self, category=None, start_date=None, end_date=None):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        df = self.df.copy()
        if category:
            df = df[df['Category'].str.lower() == category.lower()]
        if start_date:
            s = pd.to_datetime(start_date, errors='coerce')
            df = df[df['Date_parsed'] >= s]
        if end_date:
            e = pd.to_datetime(end_date, errors='coerce')
            df = df[df['Date_parsed'] <= e]
        return df

    def add_computed_columns(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        df = self.df.copy()
        df['RevenuePerItem'] = np.where((df['Quantity Sold']>0) & df['Quantity Sold'].notna(), (df['Total Sales']/df['Quantity Sold']).round(2), np.nan)
        daily = df.dropna(subset=['Date_parsed','Total Sales']).groupby('Date_parsed')['Total Sales'].sum().sort_index()
        daily = daily.reset_index().rename(columns={'Total Sales':'DailySales'})
        daily['DailySalesShift'] = daily['DailySales'].shift(1)
        daily['GrowthPct'] = ((daily['DailySales'] - daily['DailySalesShift'])/daily['DailySalesShift']*100).replace([np.inf, -np.inf], np.nan).round(2)
        self._daily_sales = daily
        self.df = df
        return df

    def plot_total_sales_by_category(self, top_n=10, save_path=None):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        agg = self.df.dropna(subset=['Category','Total Sales']).groupby('Category')['Total Sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(10,5))
        sns.barplot(x=agg.index, y=agg.values)
        plt.xlabel("Total Sales")
        plt.title("Total Sales by Category")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_sales_trend(self, save_path=None):
        if not hasattr(self, '_daily_sales'):
            _ = self.add_computed_columns()
        daily = self._daily_sales.dropna(subset=['DailySales'])
        plt.figure(figsize=(10,5))
        sns.lineplot(x='Date_parsed', y='DailySales', data=daily)
        plt.xlabel("Date")
        plt.ylabel("Daily Sales")
        plt.title("Sales Trend Over Time")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_price_quantity_heatmap(self, save_path=None):
        df = self.df.copy()
        corr_df = df[['Price','Quantity Sold','Total Sales']].corr()
        plt.figure(figsize=(6,4))
        sns.heatmap(corr_df, annot=True, fmt=".2f")
        plt.title("Correlation: Price, Quantity Sold, Total Sales")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retail Sales Data Analyzer")
    parser.add_argument("file", help="Path to retail_sales.csv")
    args = parser.parse_args()

    analyzer = RetailAnalyzer()
    analyzer.load_data(args.file)
    print("Validation issues:", analyzer.validate())
    metrics = analyzer.calculate_metrics()
    print("Metrics:", metrics)
    analyzer.add_computed_columns()

    analyzer.plot_total_sales_by_category()
    analyzer.plot_sales_trend()
    analyzer.plot_price_quantity_heatmap()
