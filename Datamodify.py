import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_excel('C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\Grass1.xlsx')

# Convert values to 1 if greater than 34, else 0
df['Label'] = df['Grass'].apply(lambda x: 1 if x > 34 else 0)

# If you want to overwrite the original column:
# df['Your_Column_Name'] = df['New_Column']

# Save the modified DataFrame to a new CSV file
df.to_excel('C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\Grass1.xlsx', index=False)
