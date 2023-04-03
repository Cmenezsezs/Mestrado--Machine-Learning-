import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from scipy.stats import multivariate_normal

class BayesianoGaussiano():

    def __init__(self):
        self.N_classes = None
        self.N_amostras = None
        self.N_features = None

        self.X_treino = None
        self.y_treino = None
        
        self.medias = None
        self.covariancia = None
        
        self.P_priori = None
        self.P_posteriori = None
               

    def fit(self, X_treino, y_treino):
        N_amostras = X_treino.shape[0]

        self.X_treino = X_treino
        self.y_treino = y_treino

        self.N_features = X_treino.shape[1]
        self.N_classes = np.max(self.y_treino)+1
        
        self.medias = np.zeros((self.N_features, self.N_classes))
        self.covariancia = np.zeros((self.N_features, self.N_features, self.N_classes))

        # Probabilidade a priori das classes
        self.P_priori = np.array([
            np.where(classe == self.y_treino)[0].shape[0]/N_amostras 
            for classe in range(self.N_classes)
        ])

        # Parâmetros das médias e covariâncias das features
        for classe in range(self.N_classes):
            ind = np.where(y_treino == classe)[0]
            X_classe = X_treino[ind,:]

            self.medias[:,classe] = np.mean(X_classe, axis=0)
            cov = np.diag(np.diag(np.cov(X_classe.T)))
            for xx in range(cov.shape[0]):
                if cov[xx, xx] == 0:
                    aux_1 = (1/(self.N_features*len(X_classe)))
                    aux_2 = np.linalg.norm((X_classe - self.medias[:, classe]), 2)**2
                    cov[xx, xx] = aux_1 * aux_2
            self.covariancia[:,:,classe] = cov
    

    def predict(self, P_teste):
        previsoes = np.argmax(P_teste, axis=1)
        return previsoes


    def predict_proba(self, X_teste):
        N_amostras = X_teste.shape[0]
        self.P_posteriori = np.zeros((N_amostras, self.N_classes))

        for amostra in range(N_amostras):

            for classe in range(self.N_classes):
                self.P_posteriori[amostra,classe] = multivariate_normal.pdf(
                    X_teste[amostra,:], 
                    mean=self.medias[:,classe],
                    cov=self.covariancia[:,:,classe]
                ) * self.P_priori[classe]
            self.P_posteriori[amostra,:] = self.P_posteriori[amostra,:]/np.sum(self.P_posteriori[amostra,:])

        return self.P_posteriori



class BayesianoParzen():

    def __init__(self, h):
        self.h = h
        self.N_classes = None
        
        self.X_treino = None
        self.y_treino = None
        
        self.P_priori = None
        self.estimadores = []

    def fit(self, X_treino, y_treino):
        N_amostras = X_treino.shape[0]
        self.N_classes = np.max(y_treino)+1

        self.X_treino = X_treino
        self.y_treino = y_treino

        for classe in range(self.N_classes):
            indices = np.where(self.y_treino == classe)
            modelo = KernelDensity(kernel='gaussian', bandwidth=self.h)
            modelo.fit(self.X_treino[indices])
            self.estimadores.append(modelo)

        self.P_priori = np.array([
            np.where(classe == self.y_treino)[0].shape[0]/N_amostras 
            for classe in range(self.N_classes)
        ])
    
    def predict_proba(self, X_teste):
        N_amostras = X_teste.shape[0]
        P_verossimilhanca = np.zeros((N_amostras, self.N_classes))
        
        for classe, estimador in enumerate(self.estimadores):
            P_verossimilhanca[:,classe] = np.exp(estimador.score_samples(X_teste)) * \
                self.P_priori[classe]

        denominador = np.sum(P_verossimilhanca, axis=1)
        denominador = np.where(denominador == 0, 1, denominador)
        self.P_posteriori = np.array(
            [P_verossimilhanca[amostra,:]/denominador[amostra] 
            for amostra in range(N_amostras)]
        )
        return self.P_posteriori

    def predict(self, P_teste):
        return np.argmax(P_teste, axis=1)

class RegressaoLogistica():

    def __init__(self):
        self.N_classes = None
        self.N_features = None

        self.X_treino = None
        self.y_treino = None
    
    def fit(self, X_treino, y_treino):
        self.X_treino = X_treino
        self.y_treino = y_treino
        self.N_classes = 1+np.max(y_treino)
        self.N_features = X_treino.shape[1]
        N_amostras = X_treino.shape[0]
        M_treino = np.zeros((N_amostras, self.N_classes))
        for classe in range(self.N_classes):
            ind = np.where(classe == self.y_treino)[0]
            M_treino[ind,classe] = 1
        self.classificador = OneVsRestClassifier(LogisticRegression()).fit(self.X_treino, M_treino)
        
    def predict_proba(self,X_teste):
        return self.classificador.predict_proba(X_teste)


    def predict(self, P_teste):
        return np.argmax(P_teste, axis=1)


class VotoMajoritario():

    def __init__(self):
        self.P_posteriori = None
    
    def predict_proba(self, P_posteriori):
        self.P_posteriori = P_posteriori
        previsoes = np.zeros((self.P_posteriori.shape[0],self.P_posteriori.shape[1]))

        for amostra in range(self.P_posteriori.shape[0]):
            argumento = np.array([])
            for dataset in range(self.P_posteriori.shape[2]):
                argumento = np.argmax(self.P_posteriori[amostra, :, dataset])
                previsoes[amostra,argumento] += 1
        return previsoes/self.P_posteriori.shape[2]
    
    def predict(self, P_teste):
        return np.argmax(P_teste, axis=1)
