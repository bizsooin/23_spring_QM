from lang_detect import cleaning

import numpy as np
import pandas as pd
import re
import os

df = pd.read_csv('review_most.csv')

# Data cleaning
df_clean = cleaning(df)

# Erase duplicated rows
df_cleanx = df_clean[~df_clean['text'].duplicated(keep='last')]

# Reset index
df_clean = df_cleanx.reset_index(drop=True)

# Save the cleaned data
df_clean.to_csv('cleaned_data.csv', index=False)