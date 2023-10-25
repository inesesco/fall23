import pandas as pd
import matplotlib.pyplot as plt

# Load your Excel data into a DataFrame
df = pd.read_excel('Gender and Fisc Yr All.xlsx')

# Group the data by 'INCR_FISC_YR' and 'GENDER'
grouped = df.groupby(['INCR_FISC_YR', 'GENDER'])

# Get the unique conditions from the column names (excluding 'INCR_FISC_YR' and 'GENDER')
conditions = [col for col in df.columns if col not in ['INCR_FISC_YR', 'GENDER']]

# Create a line plot for each condition for all fiscal years
for condition in conditions:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for year, group in grouped:
        print(year, group)
        label = f'Fiscal Year {year[0]}, Gender {year[1]}'
        data = group[['INCR_FISC_YR', 'GENDER', condition]]
        data = data.pivot(index='INCR_FISC_YR', columns='GENDER', values=condition)
        data.plot(kind='line', ax=ax, label=label)

    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Count')
    ax.set_title(f'{condition} Over Time by Gender')
    plt.grid(True)
    
    # Show the plot or save it to a file
    # plt.show()
