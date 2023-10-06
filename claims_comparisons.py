import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel data into a DataFrame
df = pd.read_excel('Total Claims Counts - Language.xlsx')

# Identify the condition columns (assuming they start from the 3rd column, adjust if needed)
claims_categories = df.columns[2:]

# Filter out languages with less than 32 members and not equal to "Total"
df_filtered = df[(df['Primary Language Count'] >= 32) & (df['PRIMARYLANGUAGE'] != "NULL")]

# Define a function to calculate the threshold based on English speakers' percentage
def calculate_threshold(english_percentage):
    return 0.5 * english_percentage

#convert to percentages
for category in claims_categories:
    df_filtered[f'{category} per person'] = (df_filtered[category] / df_filtered['Primary Language Count'])

claims_pp_categories = df_filtered.columns[2+len(claims_categories):]
print(df_filtered)

# Perform calculations and create box plots for each condition
for category in claims_pp_categories:
    # Filter the DataFrame for "Other Languages" and English speakers for each condition
    english_speakers_claims = df_filtered.loc[df_filtered['PRIMARYLANGUAGE'] == 'ENGLISH', category]
    other_languages_claims = df_filtered.loc[df_filtered['PRIMARYLANGUAGE'] != 'ENGLISH', category]

    # Calculate the observed difference in percentages
    observed_diff = english_speakers_claims.item() - np.mean(other_languages_claims)

    # Perform permutation test
    num_permutations = 32  # Number of permutations
    perm_diffs = np.zeros(num_permutations)  # Array to store the permuted differences

    # Concatenate English and non-English percentages
    all_percentages = np.concatenate((np.array([english_speakers_claims.item()]), other_languages_claims))

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
    print(f"Mean observed difference in claims per person ({category}):", observed_diff)
    print(f"p-value ({category}):", p_value)

    # Calculate the threshold based on English speakers' percentage
    threshold = calculate_threshold(english_speakers_claims.item())

    # Filter languages with a significant difference from English
    significant_languages = df_filtered[abs(df_filtered[category]) >= abs(threshold)]
    
    # Create subplots with box plots side by side for each condition
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[other_languages_claims, english_speakers_claims],
                flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'),
                color="pink")
    plt.xlabel('Language Group')
    plt.ylabel(category)
    plt.title(f'{category}: English Speakers vs. LEP ({p_value})')
    plt.xticks([0, 1], ['Other Languages', 'English Speakers'])

    # # Add text labels for languages with a significant difference from English
    # significant_languages_list = []
    # for i, row in significant_languages.iterrows():
    #     significant_languages_list.append((row['PRIMARYLANGUAGE'], row[category]))
    # print(f"\nLanguages with the highest difference in {category} compared to English speakers:")
    # for language, difference in significant_languages_list:
    #     if language != 'ENGLISH':
    #         print(f"{language}: {difference:.2f}")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
