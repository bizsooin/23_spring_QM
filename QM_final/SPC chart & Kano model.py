import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

df = pd.read_csv('matrix_v.csv')

########## 1. SPC Chart for each Quality Dimension (Pos/Neg) ##########

fig, axs = plt.subplots(3, 6, figsize=(90, 30))  # 3 rows, 6 columns

for i, column in enumerate(df.columns):
    mean = df[column].mean()
    std = df[column].std()

# Create control limits of 2 standard deviations from the mean(L = 2)
    upper_limit = mean + 2 * std
    lower_limit = mean - 2 * std

    # Calculate the row and column indices in the grid
    row = i % 3
    col = i // 3

    # Create the SPC chart using matplotlib
    ax = axs[row, col]
    ax.plot(df.index, df[column], label='Value')
    ax.axhline(y=mean, color='gray', linestyle='--', label='Mean')
    ax.axhline(y=upper_limit, color='g', linestyle='--', label='Upper Control Limit')
    ax.axhline(y=lower_limit, color='r', linestyle='--', label='Lower Control Limit')
    ax.legend()
    ax.set(title=f'SPC Chart of {column}', xlabel='Date', ylabel='Value')

    # Remove x-axis tick labels for all but the bottom row
    if row < 3:
        ax.set_xticklabels([])

plt.tight_layout()

# Show the grid of SPC charts
plt.show()


########## 2. Kano Chart based on each time period ##########

for index, row in df.iterrows():
    # Extract the positive and negative values for the current row
    x_values = row[[column for column in df.columns if column.endswith('_pos')]]
    y_values = row[[column for column in df.columns if column.endswith('_neg')]]

    # Create the scatter plot
    plt.scatter(x_values, y_values)

    # Add labels and title to the plot
    plt.xlabel('Positive')
    plt.ylabel('Negative')
    plt.title(f'Positive vs Negative - Time period {index}')

    # Add gridlines
    plt.grid(True)

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Add labels to each point
    labels = [column.replace('_pos', '') for column in x_values.index]
    texts = []
    for label, x, y in zip(labels, x_values, y_values):
        texts.append(plt.text(x, y, label, ha='center', va='center'))
    
    # Adjust the positions of labels to prevent overlap
    adjust_text(texts)

    # Show the plot for each row
    plt.show()
