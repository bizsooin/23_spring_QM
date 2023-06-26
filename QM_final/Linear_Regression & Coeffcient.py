import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from adjustText import adjust_text

df = pd.read_csv('quality_dimension.csv')

##### 1. Sort by date and add year, month columns #####

#Sort Quality_dimension by date
df_sorted = df.sort_values('date', ascending=True)

#Convert date to datetime format
df_sorted['date'] = pd.to_datetime(df_sorted['date'])

#Add year and month columns based on datetime
df_sorted['year'] = df_sorted['date'].dt.year
df_sorted['month'] = df_sorted['date'].dt.month

#Group by year and month
start_year = 2007
end_year = 2021
step = 2

#Erase the data before 2007 and after 2021
df_filtered = df_sorted[(df_sorted['year'] >= 2007) & (df_sorted['year'] <= 2021)]

# Create the sliding year ranges
year_ranges = [(year, year + 1) for year in range(start_year, end_year)]

# Assign set numbers to each year range
bin_edges = [year_ranges[0][0]-1] + [yr[1] for yr in year_ranges]

# Create Set column based on Year
df_filtered['set'] = pd.cut(df_filtered['year'], bins=bin_edges, labels=range(1, len(year_ranges) + 1))


##### 2. Create a bar chart for each set #####

set_counts = df_filtered['year'].value_counts()
# Create a bar chart for each set
plt.bar(set_counts.index, set_counts.values)

# Customize the chart labels and title
plt.xlabel('Set')
plt.ylabel('Count')
plt.title('Occurrences of Sets')
plt.show()


##### 3. Create overall linear regression model #####

column_to_exclude = ['stars', 'date', 'year', 'month','user_id']
colnames = df_sorted.columns.drop(column_to_exclude)


y = df_sorted['stars']
X = df_sorted.drop(column_to_exclude, axis=1)

model = sm.OLS(y, X)

results = model.fit()
coef = pd.DataFrame(results.params.values)

coef = coef.T
coef.columns = colnames

row_index = 0  # Set the desired row index here

# Extract the positive and negative values for the selected row
x_values = coef.iloc[row_index][[column for column in coef.columns if column.endswith('_pos')]]
y_values = coef.iloc[row_index][[column for column in coef.columns if column.endswith('_neg')]]

# Create the scatter plot
plt.scatter(x_values, y_values)

# Add labels and title to the plot
plt.xlabel('Positive')
plt.ylabel('Negative')
plt.title(f'Positive vs Negative - Overall')

# Add gridlines
plt.grid(True)

# Add abline at x=0 and y=0
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Show the plot
labels = [column.replace('_pos', '') for column in x_values.index]
texts = []
for label, x, y in zip(labels, x_values, y_values):
    texts.append(plt.text(x, y, label, ha='center', va='center'))

# Adjust the positions of labels to prevent overlap
adjust_text(texts)

# Show the plot
plt.show()


##### 4. Create correlation table higher than 0.35 absolute value #####

correlation_matrix = X.corr()
num_rows, num_cols = correlation_matrix.shape

for i in range(num_rows):
    for j in range(i+1, num_cols):
        correlation = correlation_matrix.iloc[i, j]
        if abs(correlation) > 0.35:
            row_name = correlation_matrix.index[i]
            col_name = correlation_matrix.columns[j]
            print((row_name, col_name), "Correlation:", correlation)


##### 5. Perform linear regression on each set's subset #####

reg_coef = []
grouped = df_filtered.groupby('set')

for set_name, set_df in grouped:
    # Separate the features and target variable
    column_to_exclude = ['stars', 'date', 'year', 'month', 'set', 'user_id']

    X = set_df.drop(column_to_exclude, axis=1)    

    y = set_df['stars']  # Specify the target variable column

    model = sm.OLS(y, X)

    # Fit the OLS model
    results = model.fit()

    reg_coef.append(results.params.values)


# Create Coefficient DataFrame
matrix_v = pd.DataFrame(reg_coef)

# Assign column names to the DataFrame
matrix_v.columns = colnames

matrix_v.to_csv('matrix_v.csv', index=False)


