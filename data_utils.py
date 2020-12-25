import pandas as pd
from sklearn import preprocessing

class DataUtils():

    def load_data(self, dataset_path):
        """ Creates the dataset from a database.

        Parameters
        ----------
        database_filepath: string
            Path to the database for importing the data.

        X: DataFrame
            Dataset features.
        y: DataFrame
            Dataset targets (Categories).

        category_names: list
            List of category names.
        """
        le = preprocessing.LabelEncoder()
        df = pd.read_csv(dataset_path)
        df['le'] = le.fit_transform(df['airline_sentiment'])

        X = df['text']
        y = df[['le']]

        labels = df[['airline_sentiment', 'le']].drop_duplicates()
        labels.set_index('le', inplace=True)
        category_names = list(labels.to_dict()['airline_sentiment'].values())
        return X, y, category_names

    def get_encodings(self,filepath):
        """ Prints encodings related with a given file

        Parameters
        ----------
        filepath: string
            Path to the file to analyse.
        """
        from encodings.aliases import aliases

        alias_values = set(aliases.values())

        for encoding in set(aliases.values()):
            try:
                df=pd.read_csv(filepath, encoding=encoding)
                print('successful', encoding)
            except:
                pass


    def save_db_data(self, df, database_filename):
        """Stores the processed message's dataframe

        Parameters
        ----------
        df: DataFrame
            DataFrame to store.

        database_filename: string
            Name of the database to store the data.

        """
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///'+database_filename)
        df.to_sql('messages', engine, index=False, if_exists='replace')        