# ## This projects used the following dataset: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
# 
# ## It consists of two CSV files which will be joined together in this code
# 
# ## The goal of this analysis is to find interesting insights to visualize about books and how people review them


# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Changing max rows displayed with pandas
pd.set_option('display.max_rows', 40)


df = pd.read_csv("books_data.csv")
df2 = pd.read_csv("Books_rating.csv")

df["Title"] = df["Title"].str.lower().str.strip()
df["authors"] = df["authors"].str.lower().str.strip()
df2["Title"] = df2["Title"].str.lower().str.strip()


# ## Merging the two datasets

merged = df.merge(df2, how="inner", on="Title")


# ## Check distribution of review scores

# Group and count the values
score_counts = merged.groupby("review/score")["Title"].count().reset_index()
score_counts.columns = ['Review Score', 'Count']

# Create a Seaborn bar plot
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Review Score', y='Count', data=score_counts, palette="viridis")

ax.set_title("Distribution of Review Scores")
ax.set_xlabel("Review Score")
ax.set_ylabel("Amount of reviews (millions)")

plt.tight_layout()
plt.show()

merged['review/time'] = pd.to_datetime(merged['review/time'], unit='s')

merged

df = merged.copy()

df = df.reset_index()

# Remove rows where the review time is wrong (dates before the official first day)
df = df.loc[df["review/time"] > "1995-01-01"]


df.set_index('review/time', inplace=True)


# ## Review scores over time


# Resample data by month and compute average review score
monthly_avg = df.resample('Y').mean()

# Plot
plt.figure(figsize=(12,6))
plt.plot(monthly_avg.index, monthly_avg['review/score'], marker='o')
plt.title('Average Review Score Over Time')
plt.xlabel('Time')
plt.ylabel('Average Review Score')
plt.grid(True)
plt.tight_layout()
plt.show()

df.reset_index(inplace=True)


df.columns


# ## Adding and changing columns


df['helpful_votes'], df['total_votes'] = zip(*df['review/helpfulness'].str.split('/').tolist())
df['helpful_votes'] = df['helpful_votes'].astype(int)
df['total_votes'] = df['total_votes'].astype(int)

# Calculate the helpfulness ratio
df['helpfulness_ratio'] = df['helpful_votes'] / df['total_votes']

# Handle potential NaN values (when total_votes is 0)
df['helpfulness_ratio'].fillna(0, inplace=True)


df["Title"] = df["Title"].str.lower().str.strip()


# ## Create a limited DF books with at least 50 reviews


book_stats = df.groupby('Title').agg(
    average_rating=('review/score', 'mean'),
    review_count=('review/score', 'size'),
    avg_helpfulness=('helpfulness_ratio', 'mean'),
    rating_std_dev=('review/score', 'std')
)

# Filter out books with fewer than 50 reviews
filtered_books = book_stats[book_stats['review_count'] >= 50]

# Sort books by average rating, for example
sorted_books = filtered_books.sort_values(by='average_rating', ascending=False)

sorted_books.reset_index(inplace=True)

sorted_books

sorted_books.reset_index(inplace=True)


# ## Finding the most controversial books (high standard deviation of review scores)

sorted_by_std = sorted_books.sort_values(by='rating_std_dev', ascending=False)

# Top 5
top_5_divisive = sorted_by_std.head(5)

colors = plt.cm.viridis(np.linspace(0, 1, len(top_5_divisive)))

plt.figure(figsize=(10, 5))
bars = plt.bar(range(len(top_5_divisive)), top_5_divisive['rating_std_dev'], color=colors)

# Attach a label (title) to each bar for the legend
for bar, title in zip(bars, top_5_divisive['Title'].str.capitalize()):
    bar.set_label(title)

plt.ylabel('Standard Deviation of Ratings')
plt.title('Top 5 Most Divisive Books')
plt.xticks([])  # This removes the x-axis tick labels
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, title="Book Titles")  # Places legend outside the plot
plt.tight_layout()
plt.show()


df.set_index('review/time', inplace=True)


# ## Examine amount of reviews over time

monthly_reviews = df.resample('M').size()  # Monthly resample with 'M'

# Plot
plt.figure(figsize=(12,6))
monthly_reviews.plot(kind='line', color='dodgerblue')
plt.title('Number of Reviews Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Does christmas lead to spikes in reviews?

monthly_reviews = df.resample('M').size()

# Extract points corresponding to every December
christmas_reviews = monthly_reviews[monthly_reviews.index.month == 12]

# Plot
plt.figure(figsize=(12,6))
monthly_reviews.plot(kind='line', color='dodgerblue', linewidth=1.5, label='Monthly Reviews')  # Adjusted line width
plt.scatter(christmas_reviews.index, christmas_reviews.values, color='red', s=15, label='Christmas')  # Adjusted dot size
plt.title('Number of Reviews Over Time with Christmas Highlighted')
plt.xlabel('Time')
plt.ylabel('Number of Reviews')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Violin chart of helpfulness review ratios (are the opinions of reviews often split or not)

sns.set_style("whitegrid")
sns.set_context("talk")

# Create the plot
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x=df['helpfulness_ratio'], spanmode = 'hard', inner="quartile", palette="viridis")

# Adding titles and labels
plt.title('Distribution of Helpfulness Ratio', fontsize=16)
plt.xlabel('Helpfulness Ratio', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Control the width of the plot for better visibility
ax.set_xlim(-0.1, 1.1)

# Display the plot
plt.tight_layout()
plt.show()


# ## Filter those with less than 5 reviews, can scew the chart

filtered_df = df[df['total_votes'] >= 5]

# Create the plot using the filtered dataframe
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x=filtered_df['helpfulness_ratio'], spanmode='hard', inner="quartile", palette="viridis")

# Adding titles and labels
plt.title('Distribution of Helpfulness Ratio (with more than 5 votes)', fontsize=16)
plt.xlabel('Helpfulness Ratio', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Control the width of the plot for better visibility
ax.set_xlim(-0.1, 1.1)

# Display the plot
plt.tight_layout()
plt.show()