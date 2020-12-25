import warnings
import click

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from data_utils import DataUtils
from model_utils import ModelUtils
from nlp_utils import NLPUtils
import mlflow

mlflow.sklearn.autolog()
warnings.filterwarnings('ignore')

modelUtils = ModelUtils()
dataUtils = DataUtils()
nlpUtils = NLPUtils()


@click.command()
@click.option('--dataset_path', help='Dataset path')
def load_data(dataset_path):
    return None


def pre_process(X):
    X = nlpUtils.create_vector_model(X)
    return nlpUtils.normalize_count_vector(X)


def build_model(X, y, category_names):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    model = modelUtils.build_model()
    model.fit(X_train, Y_train)
    modelUtils.evaluate_model(model, X_test, Y_test, category_names)


'''
def save_model():
    modelUtils.save_model(model, model_filepath)
    next(end)
'''

if __name__ == '__main__':
    print('Main')
    X, y, category_names = dataUtils.load_data('dataset/tweets_public.csv')
    print(y)
    #X = pre_process(X)
    print('Data preprocesed')
    with mlflow.start_run():
        build_model(X, y, category_names)

