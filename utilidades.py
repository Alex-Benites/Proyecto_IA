"""
Utilidades compartidas entre scripts.

Se importa desde 09_modelo_mlp.py, 05_graficos.py y 06_probar_modelo.py.
(Los scripts numerados no se pueden importar entre si porque sus nombres
empiezan con digito, asi que lo comun vive aqui.)
"""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class ModeloEtiquetado(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador para que trabaje con etiquetas de TEXTO.

    ¿Por que existe?
    ----------------
    scikit-learn 1.8.0 tiene un bug en MLPClassifier: con early_stopping=True
    ejecuta np.isnan() sobre las predicciones, y eso revienta si las etiquetas
    son texto:

        TypeError: ufunc 'isnan' not supported for the input types

    La solucion es entrenar con enteros (ARMA_FUEGO -> 0, etc.) y decodificar
    al predecir. Este envoltorio hace exactamente eso, de forma transparente:
    por fuera se comporta como cualquier clasificador de sklearn que recibe y
    devuelve texto, asi que 05_graficos.py y 06_probar_modelo.py lo usan igual
    que a los demas modelos.

    NO cambia el modelo: solo cambia como se representan las etiquetas
    internamente. Los pesos aprendidos son los mismos.

    Cumple el contrato de sklearn (get_params/set_params via BaseEstimator),
    asi que clone() funciona y puede usarse dentro de curvas de aprendizaje
    o permutation_importance.
    """

    def __init__(self, modelo=None):
        self.modelo = modelo

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.modelo_ = clone(self.modelo)
        self.modelo_.fit(X, self.le_.transform(y))
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.modelo_.predict(X))

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)

    # --- Acceso a los atributos del modelo interno (loss_curve_, coefs_, ...) ---
    def __getattr__(self, nombre):
        if nombre.startswith("_") or nombre in ("modelo", "modelo_", "le_"):
            raise AttributeError(nombre)
        try:
            return getattr(self.__dict__["modelo_"], nombre)
        except KeyError:
            raise AttributeError(nombre) from None
