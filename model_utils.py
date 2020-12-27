from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from nlp_utils import NLPUtils
from mlflow import log_metric


class ModelUtils():

    def build_model(self):
        """ Builds the pipeline and finds the best classification model with gridsearch.

        Returns
        -------

        cv: GridSearchCV
            GridSearchCV instance with the tuned model.
        """
        nlpUtils = NLPUtils()
        model = RandomForestClassifier()
        pipeline = Pipeline([
             ('vect', CountVectorizer(tokenizer=nlpUtils.tokenize)),
             ('tfidf', TfidfTransformer()),
             ('dense_transformer', DenseTransformer()),
             ('clf', model)
        ])

        parameters = {
            'clf__n_estimators': [100, 150, 200],
            'clf__class_weight': ['balanced', 'balanced_subsample']
        }

        scorer = make_scorer(accuracy_score)
        cv = GridSearchCV(pipeline, scoring=scorer, param_grid=parameters, verbose=50)

        return cv

    def evaluate_model(self, model, X_test, Y_test, category_names):
        """Model evaluation

        Parameters
        ----------

        model: GridSearchCV
            GridSearchCV instance with the tuned model.

        X_test: Series.
            Dataset with the test features (messages).

        Y_test: Series.
            Dataset with the test targets (categories).
        """
        print(Y_test['le'].unique())
        print(category_names)
        y_pred = model.predict(X_test)

        log_metric("test_f1_score", f1_score(Y_test, y_pred, average='micro'))
        log_metric("test_accuracy_score", accuracy_score(Y_test, y_pred))
        log_metric("test_precision_score", precision_score(Y_test, y_pred, average='micro'))

    def save_model(self, model, model_filepath):
        """Stores model

        Parameters
        ----------

        model: GridSearchCV
            GridSearchCV instance with the tuned model.

        model_filepath: string
            Path to store the model
        """
        import pickle
        # save the classifier
        pickle.dump(model, open(model_filepath, 'wb'))
        return True


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()