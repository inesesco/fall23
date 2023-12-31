import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel data into a DataFrame
df = pd.read_excel('condition_counts1.xlsx')

# Identify the condition columns (assuming they start from the 3rd column, adjust if needed)
condition_columns = df.columns[2:]

# Filter out languages with less than 32 members and not equal to "Total"
df_filtered = df[(df['PRIMARYLANGUAGE_COUNT'] >= 3000)]
for condition in condition_columns:
    df_filtered[f'% with {condition}'] = (df_filtered[condition] / df_filtered['PRIMARYLANGUAGE_COUNT']) * 100

# Perform calculations and create box plots for each condition
for condition in condition_columns:
    

    # Filter the DataFrame for "Other Languages" and English speakers for each condition
    english_speakers_condition = df_filtered.loc[df_filtered['PRIMARYLANGUAGE'] == 'ENGLISH', f'% with {condition}']
    print(english_speakers_condition)
    other_languages_condition = df_filtered.loc[df_filtered['PRIMARYLANGUAGE'] != 'ENGLISH', f'% with {condition}']

    sns.boxplot(data=[other_languages_condition, english_speakers_condition],
                flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'),
                color="#253532")
    plt.xlabel('Language Group')
    plt.ylabel(f'% with {condition}')
    plt.title(f'Distribution of {condition} %: English Speakers vs. LEP')
    plt.xticks([0, 1], ['Other Languages', 'English Speakers'])


    # Show the plot
    plt.show()
