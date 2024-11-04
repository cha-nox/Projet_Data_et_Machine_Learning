import matplotlib.pyplot as plt
import seaborn as sns
from local_utils import addInfo
from numpy import hstack, sqrt
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

output = ""

if __name__ == '__main__':
    # Getting the dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_data_frame = read_csv(data_url, sep = r"\s+", skiprows = 22, header = None)
    data = hstack([raw_data_frame.values[::2, :], raw_data_frame.values[1::2, :2]])
    target = raw_data_frame.values[1::2, 2]
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    boston_data_frame = DataFrame(data = data, columns = columns)

    # Spliting features and target into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        boston_data_frame,
        target,
        test_size = 0.2,
        random_state = 42
    )

    # Creating and training the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Making predictions
    y_pred = model.predict(x_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Printing model performance metrics
    output += addInfo(
        "Métriques de performance du modèle",
        f"MSE : {mse:.2f}\n" +
        f"RMSE : {rmse:.2f}\n" +
        f"R² Score : {r2:.2f}"
    )

    # Feature and coefficients displaying
    coefficients = DataFrame({'Feature' : columns, 'Coefficient' : model.coef_})
    output += addInfo(
        "Feature - Coefficients",
        coefficients.sort_values(by = 'Coefficient', ascending = False).to_string()
    )

    # Data visualiation
    sns.pairplot(
        boston_data_frame,
        hue = 'LSTAT',
        palette = 'coolwarm',
        diag_kind = 'kde'
    )
    plt.show()

    # Informations printing
    print(output)