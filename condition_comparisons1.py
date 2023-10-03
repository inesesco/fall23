import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel data into a DataFrame
df = pd.read_excel('condition_counts1.xlsx')
print('done 1')

# Identify the condition columns (assuming they start from the 3rd column, adjust if needed)
condition_columns = df.columns[2:]
print('done 2')

# Filter out languages with less than 32 members and not equal to "Total"
df_filtered = df[(df['PRIMARYLANGUAGE_COUNT'] >= 32) & (df['PRIMARYLANGUAGE'] != "Total")]
print('done 3')

# Define a function to calculate the threshold based on English speakers' percentage
def calculate_threshold(english_percentage):
    return 0.5 * english_percentage
print('done 4')

# Perform calculations and create box plots for each condition
for condition in condition_columns:
    # Filter the DataFrame for "Other Languages" and English speakers for each condition
    english_speakers_condition = df_filtered.loc[df_filtered['PRIMARYLANGUAGE'] == 'ENGLISH', condition]
    print('done 5')
    print(english_speakers_condition)
    other_languages_condition = df_filtered.loc[df_filtered['PRIMARYLANGUAGE'] != 'ENGLISH', condition]
    print('done 6')
    print(other_languages_condition)

    # Calculate the observed difference in percentages
    observed_diff = english_speakers_condition.item() - np.mean(other_languages_condition)

    # Perform permutation test
    num_permutations = 10000  # Number of permutations
    perm_diffs = np.zeros(num_permutations)  # Array to store the permuted differences

    # Concatenate English and non-English percentages
    all_percentages = np.concatenate((np.array([english_speakers_condition.item()]), other_languages_condition))

    for i in range(num_permutations):
        # Permute the labels (group assignments)
        permuted_labels = np.random.permutation(all_percentages)

        # Calculate the difference in means between permuted groups
        perm_diff = permuted_labels[0] - np.mean(permuted_labels[1:])

        # Store the permuted difference
        perm_diffs[i] = perm_diff

    # Calculate p-value as the proportion of permuted differences greater than or equal to observed difference
    p_value = np.mean(perm_diffs >= observed_diff)

    # Print the results
    print()
    print(f"Mean observed difference in percentages ({condition}):", observed_diff)
    print(f"p-value ({condition}):", p_value)

    # Calculate the threshold based on English speakers' percentage
    threshold = calculate_threshold(english_speakers_condition.item())

    # Filter languages with a significant difference from English
    significant_languages = df_filtered[abs(df_filtered[condition]) >= abs(threshold)]
    
    # Create subplots with box plots side by side for each condition
    plt.figure(figsize=(8, 6))
    plt.subplot(221)
    sns.boxplot(data=[other_languages_condition, english_speakers_condition],
                flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'),
                color="#253532")
    plt.xlabel('Language Group')
    plt.ylabel(condition)
    plt.title(f'Distribution of {condition} %: English Speakers vs. LEP')
    plt.xticks([0, 1], ['Other Languages', 'English Speakers'])

    # Add text labels for languages with a significant difference from English
    significant_languages_list = []
    for i, row in significant_languages.iterrows():
        significant_languages_list.append((row['Primary Language'], row[condition]))
    print(f"\nLanguages with the highest difference in {condition} % compared to English speakers:")
    for language, difference in significant_languages_list:
        if language != 'ENGLISH':
            print(f"{language}: {difference:.2f}")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
