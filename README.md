# 1. Import and Initialize
analyzer = RetailAnalyzer()

# 2. Load Data
try:
    analyzer.load_data('retail_sales.csv')
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)

# 3. Validation and Metrics
print("Validation Report:", analyzer.validate())
print("Key Metrics:", analyzer.calculate_metrics())

# 4. Data Transformation (Needed for Sales Trend Plot)
df_with_new_cols = analyzer.add_computed_columns()

# 5. Filtering (Example)
filtered_df = analyzer.filter_data(
    category='Electronics',
    start_date='2023-01-01'
)

# 6. Visualization
analyzer.plot_total_sales_by_category(save_path='category_sales.png')
analyzer.plot_sales_trend(save_path='sales_trend.png')

