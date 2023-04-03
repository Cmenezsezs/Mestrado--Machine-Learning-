import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import adjusted_rand_score, f1_score

np.seterr("ignore")
class EFCM_LS1():
    """
    Essa classe representa todo o modelo, não só a função objetivo.
    """

    def __init__(self, C, V, P):
        self.C = C  # Número de Clusters
        self.V = V  # Número de Features
        self.P = P  # Número de Amostras

    # Coleta dos centroides, baseado numa variação do Algoritmo 2
    def fit_centroide(self, X_treino, T, e, Tu, Tv, printar=True):
        self.X_treino = X_treino
        self.Tu = Tu
        self.Tv = Tv
        
        # Variação do Algorítmo 2:
        # INICIALIZAÇÃO:
        iteracao = 0
        self.Uik = np.random.dirichlet(np.ones(self.C), size=self.P) 

        Uik_aux = np.random.dirichlet(np.ones(self.C), size=self.P)

        # REPETIÇÃO:
        print(end="")
        while (iteracao <= T) and (np.nanmax(np.abs(self.Uik-Uik_aux)) >= e):

            if printar:
                print(f"Iteração: {iteracao + 1}", end="\r")
            Uik_aux = self.Uik.copy()
            iteracao += 1
            # PASSO 1: Computar a matriz de protótipos (ALGORÍTMO 1)
            self.Gkj = prototipos(X_treino, self.Uik, self.C, self.V)
            # PASSO 2: Calcular os pesos (EQUAÇÃO 26)
            self.Vkj = pesos(X_treino, self.Uik, self.Gkj, self.C, self.V, self.P, self.Tv)
            # PASSO 3: Calculo do Grau de Pertinência
            self.Uik = grau_pertinencia(X_treino, self.Vkj, self.Gkj, self.C, self.V, self.P, self.Tu)

            # Calculo do centróide:
            centroide = Vmcd(self.X_treino, self.Uik, self.C, self.V)
            if (np.nanmin(centroide) < 0.1):
                return np.nanmin(centroide)
        # print(self.Gkj)
        # print(self.Vkj)
        print(end="\r")

    # Treinamento de todo o modelo
    def fit(self, X_treino, T, e, Tu, Tv, printar=True):
        self.X_treino = X_treino
        self.Tu = Tu
        self.Tv = Tv

        # Algorítmo 2:
        # INICIALIZAÇÃO:
        iteracao = 0
        self.Uik = np.random.dirichlet(np.ones(self.C), size=self.P) 
        Uik_aux = np.random.dirichlet(np.ones(self.C), size=self.P)

        # REPETIÇÃO:
        print(end="")
        while (iteracao <= T) and (np.nanmax(np.abs(self.Uik-Uik_aux)) >= e):
            if printar:
                print(f"Iteração: {iteracao + 1}", end="\r")
            Uik_aux = self.Uik.copy()
            iteracao += 1
            # PASSO 1: Computar a matriz de protótipos (ALGORÍTMO 1)
            self.Gkj = prototipos(X_treino, self.Uik, self.C, self.V)
            # PASSO 2: Calcular os pesos (EQUAÇÃO 26)
            self.Vkj = pesos(X_treino, self.Uik, self.Gkj, self.C, self.V, self.P, self.Tv)
            # PASSO 3: Calculo do Grau de Pertinência
            self.Uik = grau_pertinencia(X_treino, self.Vkj, self.Gkj, self.C, self.V, self.P, self.Tu)
        print(end="\r")


    def partition_entropy(self):
        M = -(self.Uik * np.log(self.Uik))
        M = np.where(np.isnan(M), 0, M)
        return M.sum()/self.P

    """
    # Uma das formas de calcular a partition entropy
    def partition_entropy(self):
        breakpoint()
        soma_C = 0
        for k in range(self.C):
            soma_P = 0
            for i in range(self.P):
                aux = np.log(self.Uik[i,k])
                if np.isnan(aux):
                    aux = 0
                soma_P += self.Uik[i,k] * aux
            soma_C += soma_P
        breakpoint()
        return -soma_C/self.P
    """

    # Realiza a predição no grau :D
    def predict(self):
        P = self.X_treino.shape[0]
        Uik_teste = grau_pertinencia(self.X_treino, self.Vkj, self.Gkj, self.C, self.V, P, self.Tu)
        return crisp_partition(Uik_teste, self.C), Uik_teste


    # Ponto 3:
    def score(self, y_teste, y_predito):
        ARI = adjusted_rand_score(y_teste, y_predito)
        F_meansure = f1_score(y_teste, y_predito, average='weighted')
        return ARI, F_meansure


    # Função Objetivo
    def FuncaoObjetivo(self):
        J_objetivo = funcao_objetivo(
            self.X_treino, self.Uik, self.Vkj, self.Gkj, self.C, self.V, self.P, self.Tu, self.Tv
        )
        return J_objetivo, self.Gkj, self.Vkj
    
    def modified_part_coeficient(self):
        MPC = ((self.Uik**2).sum())/self.P
        return 1 - (self.C/(self.C-1))*(1-MPC)

    def crisp_partition(self): 
        return np.argmax(self.Uik, axis=1)

################################ FIM DA CLASSE #################################


# Cálculo dos protótipos (G) [ALGORÍTMO 1]
def prototipos(Xij, Uik, C, V):
    Gkj = np.zeros((C, V))
    for k in range(C):
        for j in range(V):
            ordem = np.argsort(Xij[:,j])
            # PASSO 1:
            # Ordenando as amostras e seus repectivos graus de pertinência
            X_ordem = Xij[ordem, j]
            U_ordem = Uik[ordem, k]

            # PASSOS 2 e 3:            
            soma = -np.sum(np.abs(U_ordem))
            contador = -1
            soma_anterior = soma

            while (soma_anterior < 0):
                contador += 1
                soma += 2 * np.abs(U_ordem[contador])
                if soma_anterior == 0 and soma == 0:
                    condicao = True
                else:
                    condicao = False 
                soma_anterior = soma     
  
            # PASSOS 4, 5 e 6:
            if condicao:  # PASSO 5
                Gkj[k, j] = (X_ordem[contador] + X_ordem[contador-1])/2 # PASSO 6
            else:
                Gkj[k, j] = X_ordem[contador-1]  # PASSO 4
         
    return Gkj


# Cálculo dos pesos e relevância (V) [Equação 26]
def pesos(Xij, Uik, Gkj, C, V, P, Tv):
    Vkj = np.zeros((C, V))
    for k in range(C):
        for j in range(V):
            # Numerador:
            num_aux = 0
            for i in range(P):
                num_aux += Uik[i, k] * np.abs(Xij[i, j] - Gkj[k, j])
            numerador = np.exp(-num_aux/Tv)

            # Denominador:
            denominador = 0
            for w in range(V):
                num_aux = 0
                for i in range(P):
                    num_aux += Uik[i, k] * np.abs(Xij[i, w]-Gkj[k, w])
                denominador += np.exp(-num_aux/Tv)
            if np.isnan(numerador/denominador):
                Vkj[k, j] = 1e-10
            else:
                Vkj[k, j] = numerador/denominador
    return Vkj


# Cálculo dos graus de pertinência [Equação 27]
def grau_pertinencia(Xij, Vkj, Gkj, C, V, P, Tu):
    Uik = np.zeros((P, C))
    for k in range(C):
        for i in range(P):
            # Numerador:
            num_aux = 0
            for j in range(V):
                num_aux += Vkj[k, j] * np.abs(Xij[i, j] - Gkj[k, j])
            numerador = np.exp(-num_aux/Tu)

            # Denominador:
            denominador = 0
            for w in range(C):
                num_aux = 0 
                for j in range(V):
                    num_aux += Vkj[w, j] * np.abs(Xij[i, j] - Gkj[w, j])
                denominador += np.exp(-num_aux/Tu)
            if np.isnan(numerador/denominador):
                Uik [i, k] = 1e-10
            else:
                Uik [i, k] = numerador/denominador
    return Uik


# Calculo dos Centróides:
def Vmcd(Xij, Uik, C, V):
    Ck = np.zeros((C, V))
    # Numerador
    for k in range(Uik.shape[1]):
        numerador = 0
        denominador = 0
        for i in range(Xij.shape[0]):
            numerador += (Uik[i,k] ** 2) * Xij[i,:]
            denominador += Uik[i,k] ** 2
        Ck[k,:] =  numerador/denominador

    VMCD = np.zeros((C, C))
    
    for i in range(C):
        for j in range(C):
            if i == j:
                VMCD[i,j] = np.nan
            else:
                VMCD[i,j] = (np.linalg.norm(Ck[i,:] - Ck[j,:])) ** 2
    return VMCD


# Cálculo da função objetivo
def funcao_objetivo(Xij, Uik, Vkj, Gkj, C, V, P, Tu, Tv): # Equação 12
    soma_C = 0
    for k in range(C):
        soma_P = 0
        for i in range(P):
            soma_V = 0
            for j in range(V):
                soma_V += Vkj[k, j] * np.abs(Xij[i, j] - Gkj[k, j])
            soma_P += Uik[i, k] * soma_V
        soma_C += soma_P
    if np.isnan(soma_C):
        parte_1 = 0
    else:
        parte_1 = soma_C

    # PARTE 2:
    soma_C = 0
    for k in range(C):
        soma_P = 0
        for i in range(P):
            soma_P += Uik[i, k] * np.log(Uik[i, k])
        soma_C += soma_P
    if np.isnan(soma_C):
        parte_2 = 0
    else:
        parte_2 = Tu * soma_C

    # PARTE 3:
    soma_C = 0
    for k in range(C):
        soma_V = 0
        for j in range(V):
            soma_V += Vkj[k, j] * np.log(Vkj[k, j])
        soma_C += soma_V
    if np.isnan(soma_C):
        parte_3 = 0
    else:
        parte_3 = Tv * soma_C
    return parte_1 + parte_2 + parte_3


# Calculo da Partição Crisp
def crisp_partition(Uik, C): 
    return np.argmax(Uik, axis=1)