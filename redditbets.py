import pandas as pd
import praw
import os
from datetime import datetime
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.stem.porter import PorterStemmer


# make these secure
client_id = "4R7bdn2_mMOixA"
client_secret = "1SzQnboZARdSDGUAsm7qfdBcAE6jyg"
user_agent = "Mozilla/5.0"


def get_submissions(subreddits, sub_limit):
    """

    Get the top submissions on each Subreddit as a dataframe

    :param subreddits: List of subreddit names to read using
    :param sub_limit: The number of submissions to read from each subreddit, max is 1000
    :return: A Dataframe of the submissions.
        Indexed by Submission ID, contains the Title, Text, Subreddit Name, and Ranking for each submission.

    """
    # connect to reddit
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    df_submissions = pd.DataFrame(columns = ["title", "text", "subreddit", "hot_rank"])
    for sub_name in subreddits:
        sub = reddit.subreddit(sub_name)
        submission_titles = dict()
        submission_text = dict()
        submission_rank = dict()
        for i, submission in enumerate(sub.hot(limit=sub_limit)):
            submission_titles[submission.id] = submission.title
            submission_text[submission.id] = submission.selftext
            submission_rank[submission.id] = i

        df = pd.concat([pd.Series(submission_text, name="text"),
                        pd.Series(submission_titles, name="title"),
                        pd.Series(submission_rank, name="hot_rank")
                        ], axis=1)
        df['subreddit'] = sub_name
        df_submissions = pd.concat([df_submissions, df], axis=0)
    return df_submissions


def pull_and_save_submissions(directory="submissions"):
    """

    Get the top submissions then save them to a file in a given directory

    When this function is called the Reddit API is called to gather the top 1000 submissions for each of 4 subreddits
     > subreddits = ("investing", "pennystocks", "algotrading", "wallstreetbets").
    The data is saved as a parquet file because it is a good choice for saving semi-structured data to disk.
    Each file name will indicate the data and time the data was gathered.
    :param directory: The string identifying a directory relative to the working directory where data will be saved.

    """
    df_submissions = get_submissions(subreddits=("investing", "pennystocks", "algotrading", "wallstreetbets"),
                                     sub_limit=1000)
    current_time = datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S_UTC")
    filename = "reddit_submissions_{}.parquet".format(current_time)

    if not os.path.isdir(directory):
        os.mkdir(directory)

    directory_filename = os.path.join(directory, filename)
    df_submissions.index.name = "submission_id"
    df_submissions.reset_index().to_parquet(directory_filename)


def read_submissions(directory="submissions"):
    """
    Read the saved submissions back into a dataframe, pass the name of the directory
    # TODO : edit this to read multiple files
    :param directory:
    :return: dataframe
    """
    files_in_directory = glob.glob(directory + "/*")
    submissions = pd.read_parquet(files_in_directory[0])
    return submissions

def tokenizer(text):
    """
    Tokenize by splitting on space. Used within the vectorizer.
    :param text:
    :return:
    """
    return text.split()


def tokenizer_porter(text):
    """
    Tokenize by first applying the PorterStemmer, then split on space characters.
    :param text:
    :return:
    """
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]



class DataSplitter():
    def __init__(self, data,
                 y_col="subreddit",
                 id_cols="index",
                 x_col="text_post",
                 train_test_split_vars='default'):

        if train_test_split_vars == 'default':
            train_test_split_vars = {'test_size': 0.33, 'random_state': 42}
        print(data.columns)

        data = data.set_index(id_cols)
        print(data.columns)

        self._train_test_split_vars = train_test_split_vars
        self._y = data[y_col]
        self._X = data[x_col]

    @property
    def _train_test_split(self):
        return train_test_split(self._X, self._y, **self._train_test_split_vars)

    @property
    def X_train(self):
        return self._train_test_split[0]

    @property
    def X_test(self):
        return self._train_test_split[1]

    @property
    def y_train(self):
        return self._train_test_split[2]

    @property
    def y_test(self):
        return self._train_test_split[3]


class SubredditClassificationExperiment:
    def __init__(self,
                 split_data,
                 preprocessor=None,
                 vectorizer=None,
                 estimator=None,
                 param_grid=None,
                 evaluator=None):
        self.split_data = split_data
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.param_grid = param_grid
        self.evaluator = evaluator

        pipeline = Pipeline(steps=[
            ('vect', vectorizer),
            ('clf', estimator)
        ])

        if param_grid:
            self.clf = GridSearchCV(pipeline,
                                    param_grid,
                                    scoring='accuracy',
                                    cv=5,
                                    verbose=1,
                                    n_jobs=-1)
        else:
            self.clf = pipeline

    def fit(self):
        self.clf.fit(self.split_data.X_train, self.split_data.y_train)

    def evaluate(self):
        print("Train Set Evaluation:")
        print(classification_report(y_true=self.split_data.y_train, y_pred=self.train_predictions))

        print("Test Set Evaluation:")
        print(classification_report(y_true=self.split_data.y_test, y_pred=self.test_predictions))

    @property
    def train_predictions(self):
        return self.clf.predict(self.split_data.X_train)

    @property
    def test_predictions(self):
        return self.clf.predict(self.split_data.X_test)


def main():
    # Read Data
    data = read_submissions().reset_index()\
        .pipe(pd.melt, id_vars=("index", "subreddit", "hot_rank"), var_name="text_type", value_name="text_post")

    # Split Data into Train and Test Sets
    split_data = DataSplitter(data, x_col="text_post")

    # Initialize the Objects used in this Experimental Condition
    vectorizer = TfidfVectorizer()
    estimator = LogisticRegression(solver='liblinear')
    param_grid = [{'vect__tokenizer': [tokenizer_porter],
                   'vect__max_df': [0.2, 0.3, 0.5, 0.9],
                   'vect__min_df': [1, 2],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [300.0, 1000.0, 3000.0]
                   }]

    # Initialize the Experimental Condition to run
    exp_condition_1 = SubredditClassificationExperiment(split_data,
                                                        vectorizer=vectorizer,
                                                        estimator=estimator,
                                                        param_grid=param_grid)

    # Run and view evaluation of the Experiment
    exp_condition_1.fit()
    exp_condition_1.evaluate()


