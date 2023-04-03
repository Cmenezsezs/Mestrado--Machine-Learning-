import pandas as pd
import numpy as np
from questao_1 import EFCM_LS1
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")

################################################################################
# MANIPULAÇÃO DA BASE DE DADOS
dados_treino = pd.read_csv("base-dados/segmentation.tudo")

# Índices das features/atributos das três bases de dados
base_dados = {
    "Base 1": [4, 5, 6, 7, 8, 9],
    "Base 2": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "Base 3": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
}

particoes_crisp = np.empty((len(dados_treino), 3))

# Tu e Tv encontrados para cada dataset com base nas distâncias dos centroides
HiperParametros = {
        "Base 1": (3.7525, 0.01),
        "Base 2": (1.2575, 0.01),
        "Base 3": (3.7525, 0.01)
    }
################################################################################

k = 0 # Índice dos datasets
for key, lista in base_dados.items():
    print(key, end='\n\n')
    # Plotando os gráfigos das features:
    # fig = px.line(dados_treino, x=dados_treino.index,
    # y=dados_treino.columns[lista])
    # fig.show()

    escala = StandardScaler() # Normaliza. Deixando a média 0 e o desvio 1
    encoder = LabelEncoder()
    X_treino = escala.fit_transform(dados_treino[dados_treino.columns[lista]])
    y_treino = encoder.fit_transform(dados_treino[dados_treino.columns[0]])

    # Aleatorizando os dados:
    np.random.seed(0)
    aleatorio = np.random.permutation(len(X_treino))
    X_treino = X_treino[aleatorio, :]
    y_treino = y_treino[aleatorio]

    C = 7 # Número de clusters
    V = X_treino.shape[1] # Número de Features
    P = X_treino.shape[0] # Número de amostras na base de dados

    T = 150
    e = 10**-10

    N_iniciacao = 50

    # inicialização para coleta da melhor execução
    objetivo, modified_partition, partition_entropy = [], [], []

    prototipos = np.zeros((C, V, N_iniciacao))
    relevancia = np.zeros((C, V, N_iniciacao))
    
    crisp_part = np.zeros((P,N_iniciacao))
    crisp_final = np.zeros((P,3))

    # Escolhe Tu e Tv do dataset em questão
    Tu, Tv = HiperParametros[key]

    # Começa as repetições
    for kk in range(N_iniciacao):
        print(f"Iniciação : {kk+1}", end="\r")
        print(end="\n")
        modelo = EFCM_LS1(C, V, P) # Inicializa o modelo
        modelo.fit(X_treino, T, e, Tu, Tv) # Treina o modelo

        obj, gkj, vkj = modelo.FuncaoObjetivo()
        objetivo.append(obj)
        prototipos[:,:,kk] = gkj
        relevancia[:,:,kk] = vkj

        modified_partition.append(modelo.modified_part_coeficient())
        partition_entropy.append(modelo.partition_entropy())

        crisp_part[:,kk] = modelo.crisp_partition()

    # Coleta o índice  do melhor resutlado da função objetivo
    argumento = np.argmin(objetivo)
    np.savetxt(f"Resultado3_Protótipos_{key}", prototipos[:,:, argumento])
    np.savetxt(f"Resultado3_Relevancia{key}", relevancia[:,:, argumento])
    crisp_final[:,k] = crisp_part[:,argumento]
    np.savetxt(f"Resultado3_crisp_particao{key}" , crisp_final[:,k])

    np.savetxt(f"Resultado3_MatrizConfusao_{key}", confusion_matrix(y_treino, crisp_final[:,k]))

    print(f"Resultado3_Modified Partition: {modified_partition[argumento]}")
    print(f"Resultado3_Partition Entropy: {partition_entropy[argumento]}")
    print(f"Resultado3_Adjusted Rand Score: {adjusted_rand_score(y_treino, crisp_final[:,k])}")
    print(f"Resultado3_F-Measure: {f1_score(y_treino, crisp_final[:,k], average = 'macro')}", end="\n\n")
    fig = px.line(y=objetivo)
    fig.write_html(f"Resultado3_FiguraObjetivo_{key}.html")

################################################################################
# EXECUÇÃO DUAS A DUAS

print("Dois a Dois")
for ll in [[0,1],[0,2],[1,2]]:
    np.savetxt(f"Resultado3_MatrizConfusao_{str(ll)}", confusion_matrix(crisp_final[:,ll[0]],crisp_final[:,ll[1]]))

    print(f"Resultado3_Adjusted Rand Score {str(ll)}: {adjusted_rand_score(crisp_final[:,ll[0]],crisp_final[:,ll[1]])}")
    print(f"Resultado3_F-Measure {str(ll)}: {f1_score(crisp_final[:,ll[0]],crisp_final[:,ll[1]], average = 'macro')}", end="\n\n")