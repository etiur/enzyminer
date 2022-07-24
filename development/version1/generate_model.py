git from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import RidgeClassifier as RIDGE
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.utils import shuffle


def training():
    """A function that trains the models"""
    # loading the data
    VHSE = pd.read_excel("sequences.xlsx", index_col=0, engine='openpyxl')
    print(VHSE.columns)
    Y = VHSE["label"].copy()
    X_svc = pd.read_excel("esterase_noerror.xlsx", index_col=0, sheet_name="ch2_20", engine='openpyxl')
    X_knn = pd.read_excel("esterase_noerror.xlsx", index_col=0, sheet_name="random_30", engine='openpyxl')

    if X_svc.isnull().values.any():
        X_svc.dropna(axis=1, inplace=True)
        X_svc.drop(["go"], axis=1, inplace=True)

    if X_knn.isnull().values.any():
        X_knn.dropna(axis=1, inplace=True)
        X_knn.drop(["go"], axis=1, inplace=True)

    scaling = MinMaxScaler()
    scaling_2 = MinMaxScaler()
    esterase = ['EH51(22)', 'EH75(16)', 'EH46(23)', 'EH98(11)', 'EH49(23)']
    X_svc = X_svc.loc[[x for x in X_svc.index if x not in esterase]]
    X_knn = X_knn.loc[[x for x in X_knn.index if x not in esterase]]
    Y = Y.loc[[x for x in Y.index if x not in esterase]]
    X_svc, X_knn, Y = shuffle(X_svc, X_knn, Y, random_state=20)
    transformed_x_svc = scaling.fit_transform(X_svc)
    transformed_x_svc = pd.DataFrame(transformed_x_svc)
    transformed_x_svc.index = X_svc.index
    transformed_x_svc.columns = X_svc.columns

    transformed_x_knn = scaling_2.fit_transform(X_knn)
    transformed_x_knn = pd.DataFrame(transformed_x_knn)
    transformed_x_knn.index = X_knn.index
    transformed_x_knn.columns = X_knn.columns

    # training the knn models
    knn_20 = KNN(n_neighbors=4, p=4, metric="minkowski", n_jobs=-1)
    knn_80 = KNN(n_neighbors=4, p=2, metric="minkowski", n_jobs=-1)
    knn_20.fit(transformed_x_knn, Y)
    knn_80.fit(transformed_x_knn, Y)
    joblib.dump(knn_20, "models/knn_20.pkl")
    joblib.dump(knn_80, "models/knn_90.pkl")

    # trainning svc models
    svc_20 = SVC(C=0.31, kernel="rbf", gamma=1)
    svc_80 = SVC(C=5, kernel="rbf", gamma=0.31)
    svc_20.fit(transformed_x_svc, Y)
    svc_80.fit(transformed_x_svc, Y)
    joblib.dump(svc_20, "models/svc_20.pkl")
    joblib.dump(svc_80, "models/svc_80.pkl")

    # training ridge models
    ridge_20 = RIDGE(alpha=6, random_state=0)
    ridge_40 = RIDGE(alpha=2, random_state=0)
    ridge_80 = RIDGE(alpha=0.63, random_state=0)
    ridge_40.fit(transformed_x_svc, Y)
    ridge_20.fit(transformed_x_svc, Y)
    ridge_80.fit(transformed_x_svc, Y)
    joblib.dump(ridge_20, "models/ridge_20.pkl")
    joblib.dump(ridge_80, "models/ridge_80.pkl")
    joblib.dump(ridge_40, "models/ridge_40.pkl")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    training()
