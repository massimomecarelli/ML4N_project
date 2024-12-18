
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates # to format dates on plots
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_parquet('ssh_attacks.parquet')

df_copy = df.copy()
df_copy['first_timestamp'] = pd.to_datetime(df_copy['first_timestamp'])
df_copy.set_index('first_timestamp', inplace=True)
df_result = df_copy.resample('ME').size().reset_index(name='count')
df_result

plt.figure(figsize=(10, 6))
# for each month, count the number of rows, that are the number of packets
plt.bar(df_result['first_timestamp'], df_result['count'], alpha=0.7, width=20, color='r')
# Format x-axis to show month/year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.xlabel('Timestamp')
plt.ylabel('Attacks')
plt.title(f'Number of attacks')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# distr of attacks

df_copy = df.copy()
df_copy['first_timestamp'] = pd.to_datetime(df_copy['first_timestamp'])
df_copy.set_index('first_timestamp', inplace=True)
df_daily = df_copy.resample('D').size().reset_index(name='count')
df_daily

plt.figure(figsize=(10, 6))
# for each day, count the number of rows, that are the number of packets
plt.plot(df_daily['first_timestamp'], df_daily['count'], linestyle='-', color='r')
# Format x-axis to show month/year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.xlabel('Timestamp')
plt.ylabel('Attacks')
plt.title(f'Number of attacks')
plt.grid()
plt.show()



df_copy = df.copy()
df_copy['first_timestamp'] = pd.to_datetime(df_copy['first_timestamp'])

# Extract the day of the week (0=Monday, 6=Sunday)
df_copy['day_of_week'] = df_copy['first_timestamp'].dt.dayofweek  # Numeric representation
df_copy['day_name'] = df_copy['first_timestamp'].dt.day_name()   # Full name of the day

# Group by day of the week and count occurrences
df_grouped = df_copy.groupby('day_name').size().reset_index(name='count')

# Ensure the days are ordered from Monday to Sunday
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_grouped['day_name'] = pd.Categorical(df_grouped['day_name'], categories=day_order, ordered=True)
df_grouped = df_grouped.sort_values('day_name').reset_index(drop=True)

plt.figure(figsize=(7,5))
sns.barplot(
    data = df_grouped,
    x = "day_name",
    y = "count",
    color = 'blue'
)

plt.ylabel("Number of Attacks")
plt.xlabel("Day of the Week")
plt.title("Number of total attacks per day of the week")
plt.show()




# 1.2

n_words = df['full_session'].apply(lambda session: len(session))
n_characters = df['full_session'].apply(lambda session: sum([len(word) for word in session]))
data = pd.DataFrame({"n_characters": n_characters, "n_words": n_words})



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.ecdfplot(
    data = data['n_characters'],
    log_scale=True,
    color = 'blue'
)
plt.title("ECDF characters")
plt.xlabel("Number of characters")
plt.ylabel("CDF")

plt.subplot(1,2,2)
sns.ecdfplot(
    data = data['n_words'],
    log_scale=True,
    color = 'blue'
)
plt.title("ECDF words")
plt.xlabel("Number of words")
plt.ylabel("CDF")

plt.tight_layout()
plt.show()






# 1.3
import re

def split_words(command):
    strings = [part.strip() for part in re.split(r'[;|/.]\s*|\s+', command) if part.strip()] # split the string into a list
    # remove quoted text e.g. that can be a crypto key
    filtered_strings = [s for s in strings if not (s.startswith("'") and s.endswith("'")) and not (s.startswith('"') and s.endswith('"'))]
    
    return filtered_strings

# Define a function to filter out file paths and options
def filter_commands(command_list):
    filtered = []
    for cmd in command_list:
        # Exclude command options (strings starting with '-')
        if cmd.startswith("-"):
            continue
        if not cmd.isalpha():
            cmd = ''.join([i for i in cmd if i.isalpha()])  # Exclude everything that are not letters
        if not cmd: # check if empty string
            continue
        # Add to filtered list if it's a valid command
        filtered.append(cmd)
    return filtered

commands = df["full_session"] # double brackets to create a subset Dataframe instead of a Seires

commands = commands.apply(split_words)

# strip removes any leading and trailing spaces
# split regex:
# s*: Matches any amount of spaces after the symbols to allow for the possibility of spaces being next to the delimiter.
# s+: Matches one or more spaces in between words (acting as a delimiter between words or commands).

"""
The condition if part.strip() ensures that empty strings 
(which can occur if there are extra spaces or adjacent delimiters) 
are excluded from the resulting list.
"""
commands = commands.apply(filter_commands)



# Flatten the Series into a single list
all_elements = [item for sublist in commands for item in sublist]

# Create a DataFrame with each element as a row
commands_df = pd.DataFrame({'Command': all_elements}).groupby('Command').size().reset_index(name='count').sort_values(by='count', ascending=False).reset_index(drop=True).head(10)



plt.figure(figsize=(7,5))
sns.barplot(
    data = commands_df,
    x="Command",
    y="count"
)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Words Frequencies in Sessions")
plt.tight_layout()
plt.show()





# 1.4
df_intents = df[["session_id", "first_timestamp", "Set_Fingerprint"]]
# explode() is used to transform a list-like column in a DataFrame or Series into multiple rows, effectively "exploding" the list elements into separate rows.
df_intents_list = df_intents.explode('Set_Fingerprint')
intents = df_intents_list.groupby("session_id").size().to_frame("count")

plt.figure(figsize=(7,5))
sns.ecdfplot(
    data = intents,
    legend=False
)
plt.title("ECDF intents")
plt.xlabel("Intents")
plt.ylabel("CDF")
plt.show()


intents = df_intents_list.groupby('Set_Fingerprint').size().sort_values(ascending=False).to_frame("Number_of_sessions")

plt.figure(figsize=(7,5))
sns.set_theme(style="darkgrid")
sns.barplot(intents, x="Number_of_sessions", y=intents.index, color="blue", hue_order=intents.index)
plt.title('Distribution of Intents')
plt.xlabel('Number of sessions')
plt.ylabel('Type of Intents')

plt.show()


# distribution
df_intents_list['first_timestamp'] = pd.to_datetime(df_intents_list['first_timestamp'])
# A Grouper allows the user to specify a groupby instruction for an object.
distribution = df_intents_list.groupby([pd.Grouper(key='first_timestamp', freq='D'), 'Set_Fingerprint']).size().to_frame("count").reset_index()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(
    data = distribution,
    x = "first_timestamp",
    y = "count",
    hue="Set_Fingerprint",
    legend=True
)
plt.title("Daily Distribution of Intents")
plt.xlabel("Month/Year")
plt.ylabel("Number of the intents")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))

plt.show()

plt.show()


# 1.5
word_freq = dict(zip(commands_df['Command'], commands_df['count']))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# 1.6
# commands is a df in which each row is a session and for each we have rhe list of words used
# Step 1: Convert the list of words to a single string per session
words_per_session = commands.apply(lambda x: ' '.join(x))

# Step 2: Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Step 3: Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(words_per_session)

# Step 4: Get feature names (words)
words = tfidf_vectorizer.get_feature_names_out()

# Step 5: Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=words)

# Step 6: Add session IDs for reference
tfidf_df['session_id'] = words_per_session.index

# Step 7: Reshape DataFrame to associate each word with its TF-IDF value in each session
tfidf_long_df = tfidf_df.melt(
    id_vars=["session_id"], 
    var_name="word", 
    value_name="tfidf_value"
)

# Step 8: Filter out zero TF-IDF values
tfidf_long_df = tfidf_long_df[tfidf_long_df['tfidf_value'] > 0]

# Display the result
tfidf_long_df.to_csv('tfidf.csv', index=False)



























