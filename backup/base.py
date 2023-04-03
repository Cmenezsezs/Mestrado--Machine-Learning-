import numpy as np
import random
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def normaliza_0_1(vet):
    vet -= np.min(vet)
    vet /= np.max(vet)
    return vet


# Base de dados carregada utilizando pandas
def carrega_base(dir, colunas, tipo_base, escala=None, encoder=None, shuffle=False, seed=None, plotar_grafico=False):
    # Carregando Base de Dados:
    dados = pd.read_csv(dir)

    dados["VEDGE-SD"] = dados["VEDGE-SD"].clip(upper=20)
    dados["HEDGE-SD"] = dados["HEDGE-SD"].clip(upper=150)

    # Normaliza atributos entre 0 e 1 e cria um encode para os labels
    if tipo_base == "train":
        escala = MinMaxScaler(feature_range=(-1, 1))
        # escala = StandardScaler()
        encoder = LabelEncoder()

        atrib = escala.fit_transform(dados[dados.columns[colunas]])
        labels = encoder.fit_transform(dados[dados.columns[0]])
    elif tipo_base == "test":
        if None in [escala, encoder]:
            print("Passe como parametro [escala] e [encoder]")
            exit()
        atrib = escala.transform(dados[dados.columns[colunas]])
        labels = encoder.transform(dados[dados.columns[0]])
    else:
        print("defina a base: [train, test]")
        exit()

    # Aleatorizando os dados:
    if shuffle:
        np.random.seed(seed)
        aleatorio = np.random.permutation(len(atrib))
        atrib = atrib[aleatorio, :]
        labels = labels[aleatorio]

    # Plota o valor de cada coluna para todo elemento da base
    if plotar_grafico == True:
        dataF = pd.DataFrame(atrib, columns=colunas)
        fig = px.line(dataF, x=dataF.index, y=dataF.columns, title="Dados",
                      labels={
                          "index": "Índice Amostra",
                          "value": "Valores dos atributos",
                          "variable": "Colunas"
                      })
        fig.write_html('grafico_base_dados.html')  # Salva o HTML
        fig.show()  # Abre servidor com o plot

    return atrib, labels, escala, encoder

from sklearn.model_selection import StratifiedKFold


# TODO: ver questão do test
def repeated_stratfied_kfold(N_rep, N_fold, atrib, labels):
    """
    repeated_stratfied_kfold

    Args:
        N_rep (int): número de repetições
        N_fold (int): número de folds
        atrib (list): lista das features (numpy)
        labels (list): lista dos labels (numpy)
    """
    base_final = []
    for i in range(N_rep):
        skf = StratifiedKFold(N_fold, shuffle=True)
        for atrib_idx, labels_idx in skf.split(atrib, labels):
            base_final.append([list(atrib[atrib_idx]), list(labels[labels_idx])])


if __name__ == '__main__':

    dir_treino = "/home/baltz/Documentos/Mestrado/2021-1/AM/Mestrado-AM_Chico/base_dados/segmentation.data"
    dir_teste = "/home/baltz/Documentos/Mestrado/2021-1/AM/Mestrado-AM_Chico/base_dados/segmentation.test"

    colunas_base_1 = [4, 5, 6, 7, 8, 9]
    colunas_base_2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    colunas_base_3 = colunas_base_1 + colunas_base_2

    atrib, labels, escala, encoder= carrega_base(
        dir_treino, colunas_base_3, tipo_base="train", escala=None, encoder=None,
        shuffle=True, seed=3, plotar_grafico=False)


    repeated_stratfied_kfold(30, 10, atrib, labels)