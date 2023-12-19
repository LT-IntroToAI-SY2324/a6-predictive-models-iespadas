from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    # metadata
    print(adult.metadata)

    # variable information
    print(adult.variables)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Create logistic regression model
    model = LogisticRegression(solver='lbfgs', max_iter=1000)

    # Fit the model and handle potential convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

except ImportError:
    print("Error: ucimlrepo package not found. Ensure it is installed and imported correctly.")
except Exception as e:
    print(f"An error occurred: {e}")
