import pandas as pd
import numpy as np
import questao_2 as q2
import plotly.express as px

from questao_2 import BayesianoGaussiano, BayesianoParzen
from questao_2 import RegressaoLogistica, VotoMajoritario
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from scipy.stats import sem, t, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman as nemenyi_teste

import warnings
warnings.filterwarnings("ignore")
seed = 0

################################################################################
# MANIPULAÇÃO DA BASE DE DADOS
dados_treino = pd.read_csv("base-dados/segmentation.tudo")

# Índices das features/atributos das três bases de dados
base_dados = {
    "Base 1": [4, 5, 6, 7, 8, 9],
    "Base 2": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "Base 3": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
}

# Normalizando os dados minimo e máximo fica sendo 0 e 1, respectivamente.
escala = MinMaxScaler()
encoder = LabelEncoder()

X_treino = escala.fit_transform(dados_treino[dados_treino.columns[4:]])
y_treino = encoder.fit_transform(dados_treino[dados_treino.columns[0]])

# Separando as 3 bases de dados:
X1_treino = X_treino[:,:6]
X2_treino = X_treino[:,6:]
X3_treino = X_treino

base_treino = {
    "Base 1": X1_treino,
    "Base 2": X2_treino,
    "Base 3": X3_treino
}
################################################################################

# BUSCA POR HIPER PARÂMETROS K E H

# K = []
# H = []
K = [19, 3, 7]
H = [0.01, 0.02, 0.02]

# Encontrar os Hiperparâmetros:
# for base, dataset in base_treino.items():
#     Xtreino, Xteste, ytreino, yteste = train_test_split(
#         dataset, y_treino, test_size=0.2,
#         random_state=seed, stratify=y_treino
#     )
    
#     # K-NN
#     acuraciaK = []
#     K_range = np.arange(3,100,2)
#     for k in K_range:
#         modelo = KNeighborsClassifier(k)
#         modelo.fit(Xtreino, ytreino)
#         previsaoK = modelo.predict(Xteste)
#         acuraciaK.append(accuracy_score(yteste, previsaoK))
#         K_ap.append(accuracy_score(yteste, previsaoK))
    
#     # Plotagem da acurácia em função do Hiperparâmetro K
#     # fig = px.line(x=K_range, y=acuraciaK)
#     # fig.show()
#     K.append(K_range[np.argmax(acuraciaK)])

#     # Janela Parzen
#     acuraciaH = []
#     H_range = np.arange(0.01, 1, 0.01)

#     for h in H_range:
#         modelo = BayesianoParzen(h)
#         modelo.fit(Xtreino, ytreino)
#         P_parzen = modelo.predict_proba(Xteste)
#         previsaoH = modelo.predict(P_parzen)
#         acuraciaH.append(acacuracy_score(yteste, previsaoH))
#         H_ap.append(accuracy_score(yteste, previsaoH))

#     # Plotagem da acurácia em função do Hiperparâmetro H
#     # fig = px.line(x=H_range, y=acuraciaH)
#     # fig.show()
#     H.append(H_range[np.argmax(acuraciaH)])

# dataK = pd.DataFrame(
#     np.asarray(K_ap).transpose(),
#     colums = ['base_1', 'base_2', 'base_3'])
# fig = px.line(dataK, x=K_range, y=dataK.columns, title='Busca por K',
#     labels={
#         'x': "Valor de K",
#         'value': "Acurácia",
#         "variable": "Colunas"
#     })
# fig.show()
# dataH = pd.DataFrame(
#     np.asarray(H_ap).transpose(),
#     colums = ['base_1', 'base_2', 'base_3'])
# fig = px.line(dataH, x=H_range, y=dataH.columns, title='Busca por H',
#     labels={
#         'x': "Valor de H",
#         'value': "Acurácia",
#         "variable": "Colunas"
#     })
# fig.show()

# Para a base de dados .TUDO 
print("K: ", K)
print("H: ", H)
################################################################################

# REPEATED STRATIFIED K-FOLD
n_splits = 10
n_repeats = 30

rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed
    )

################################################################################

dict_txerro = {
    "Gaussiano": [],
    "K_NN": [],
    "Parzen": [],
    "RegLog": [],
    "Combinado": []
}
dict_precisao = {
    "Gaussiano": [],
    "K_NN": [],
    "Parzen": [],
    "RegLog": [],
    "Combinado": []
}
dict_cobertura = {
    "Gaussiano": [],
    "K_NN": [],
    "Parzen": [],
    "RegLog": [],
    "Combinado": []
}
dict_fscore = {
    "Gaussiano": [],
    "K_NN": [],
    "Parzen": [],
    "RegLog": [],
    "Combinado": []
}

txerro_gauss = []
precisao_gauss = []
cobertura_gauss = []
fscore_gauss = []

txerro_knn = []
precisao_knn = []
cobertura_knn = []
fscore_knn = []

txerro_parzen = []
precisao_parzen = []
cobertura_parzen = []
fscore_parzen = []

txerro_reglog = []
precisao_reglog = []
cobertura_reglog = []
fscore_reglog = []

txerro_combinacao = []
precisao_combinacao = []
cobertura_combinacao = []
fscore_combinacao = []

contador = 0

################################################################################
# EXECUÇÃO DOS MODELOS]

for index_treino, index_teste in rskf.split(X_treino, y_treino):
    print(contador+1, end='\r')

    ytr, yte = y_treino[index_treino], y_treino[index_teste]

    P_gauss = {
        "Base 1": [],
        "Base 2": [],
        "Base 3": []
    }
    P_gauss_teste = P_gauss.copy()

    P_knn = {
        "Base 1": [],
        "Base 2": [],
        "Base 3": []
    }
    P_knn_teste = P_knn.copy()

    P_parzen = {
        "Base 1": [],
        "Base 2": [],
        "Base 3": []
    }
    P_parzen_teste = P_parzen.copy()

    P_reglog = {
        "Base 1": [],
        "Base 2": [],
        "Base 3": []
    }
    P_reglog_teste = P_reglog.copy()

    P_combinacao = {
        "Base 1": [],
        "Base 2": [],
        "Base 3": []
    }
    P_combinacao_teste = P_combinacao.copy()
    
    b = 0
    for base, dataset in base_treino.items():
        Xtr, Xte = dataset[index_treino], dataset[index_teste]
        
        # Classificador Bayesiano Gaussiano:
        classif_gauss = BayesianoGaussiano()
        classif_gauss.fit(Xtr, ytr)
        P_gauss[base] = classif_gauss.predict_proba(Xte)

        # Classificador Bayesiano K-NN:
        classif_knn = KNeighborsClassifier(K[b])
        classif_knn.fit(Xtr, ytr)
        P_knn[base] = classif_knn.predict_proba(Xte)
        

        # Classificador Bayesiano Parzen:
        classif_parzen = BayesianoParzen(H[b])
        classif_parzen.fit(Xtr,ytr)
        P_parzen[base] = classif_parzen.predict_proba(Xte)

        # Classificador Regressão Logística:
        classif_reglog = RegressaoLogistica()
        classif_reglog.fit(Xtr, ytr)
        P_reglog[base] = classif_reglog.predict_proba(Xte)

        b += 1
    
    # Modelo Bayesiano Gaussiano
    P_gauss = np.dstack((P_gauss["Base 1"], P_gauss["Base 2"], P_gauss["Base 3"]))
    voto = VotoMajoritario()
    P_gauss_voto = voto.predict_proba(P_gauss)
    previsao_gauss = voto.predict(P_gauss_voto)
    txerro_gauss.append(1-accuracy_score(yte, previsao_gauss))
    precisao_gauss.append(precision_score(yte, previsao_gauss, average='macro'))
    cobertura_gauss.append(recall_score(yte, previsao_gauss, average='macro'))
    fscore_gauss.append(f1_score(yte, previsao_gauss, average='macro'))

    P_knn = np.dstack((P_knn["Base 1"], P_knn["Base 2"], P_knn["Base 3"]))
    voto = VotoMajoritario()
    P_knn_voto = voto.predict_proba(P_knn)
    previsao_knn = voto.predict(P_knn_voto)
    txerro_knn.append(1-accuracy_score(yte, previsao_knn))
    precisao_knn.append(precision_score(yte, previsao_knn, average='macro'))
    cobertura_knn.append(recall_score(yte, previsao_knn, average='macro'))
    fscore_knn.append(f1_score(yte, previsao_knn, average='macro'))

    P_parzen = np.dstack((P_parzen["Base 1"], P_parzen["Base 2"], P_parzen["Base 3"]))
    voto = VotoMajoritario()
    P_parzen_voto = voto.predict_proba(P_parzen)
    previsao_parzen = voto.predict(P_parzen_voto)
    txerro_parzen.append(1-accuracy_score(yte, previsao_parzen))
    precisao_parzen.append(precision_score(yte, previsao_parzen, average='macro'))
    cobertura_parzen.append(recall_score(yte, previsao_parzen, average='macro'))
    fscore_parzen.append(f1_score(yte, previsao_parzen, average='macro'))

    P_reglog = np.dstack((P_reglog["Base 1"], P_reglog["Base 2"], P_reglog["Base 3"]))
    voto = VotoMajoritario()
    P_reglog_voto = voto.predict_proba(P_reglog)
    previsao_reglog = voto.predict(P_reglog_voto)
    txerro_reglog.append(1-accuracy_score(yte, previsao_reglog))
    precisao_reglog.append(precision_score(yte, previsao_reglog, average='macro'))
    cobertura_reglog.append(recall_score(yte, previsao_reglog, average='macro'))
    fscore_reglog.append(f1_score(yte, previsao_reglog, average='macro'))
    
    P_combinacao = np.dstack((P_gauss_voto, P_knn_voto, P_parzen_voto, P_reglog_voto))
    voto = VotoMajoritario()
    P_combinacao_voto = voto.predict_proba(P_combinacao)
    previsao_combinacao = voto.predict(P_combinacao_voto)
    txerro_combinacao.append(1-accuracy_score(yte, previsao_combinacao))
    precisao_combinacao.append(precision_score(yte, previsao_combinacao, average='macro'))
    cobertura_combinacao.append(recall_score(yte, previsao_combinacao, average='macro'))
    fscore_combinacao.append(f1_score(yte, previsao_combinacao, average='macro'))

    contador += 1

    if (contador % 10 == 0):
        # Adicionando as médias das acurácias para cada repetição:
        dict_txerro["Gaussiano"].append(np.mean(txerro_gauss))
        dict_txerro["K_NN"].append(np.mean(txerro_knn))
        dict_txerro["Parzen"].append(np.mean(txerro_parzen))
        dict_txerro["RegLog"].append(np.mean(txerro_reglog))
        dict_txerro["Combinado"].append(np.mean(txerro_combinacao))

        # Adicionando as médias das precisões para cada repetição:
        dict_precisao["Gaussiano"].append(np.mean(precisao_gauss))
        dict_precisao["K_NN"].append(np.mean(precisao_knn))
        dict_precisao["Parzen"].append(np.mean(precisao_parzen))
        dict_precisao["RegLog"].append(np.mean(precisao_reglog))
        dict_precisao["Combinado"].append(np.mean(precisao_combinacao))

        # Adicionando as médias das coberturas para cada repetição:
        dict_cobertura["Gaussiano"].append(np.mean(cobertura_gauss))
        dict_cobertura["K_NN"].append(np.mean(cobertura_knn))
        dict_cobertura["Parzen"].append(np.mean(cobertura_parzen))
        dict_cobertura["RegLog"].append(np.mean(cobertura_reglog))
        dict_cobertura["Combinado"].append(np.mean(cobertura_combinacao))

        # Adicionando as médias dos F-Scores para cada repetição:
        dict_fscore["Gaussiano"].append(np.mean(fscore_gauss))
        dict_fscore["K_NN"].append(np.mean(fscore_knn))
        dict_fscore["Parzen"].append(np.mean(fscore_parzen))
        dict_fscore["RegLog"].append(np.mean(fscore_reglog))
        dict_fscore["Combinado"].append(np.mean(fscore_combinacao))
        
        # Resetando para os Folds
        txerro_gauss = []
        precisao_gauss = []
        cobertura_gauss = []
        fscore_gauss = []
        
        txerro_knn = []
        precisao_knn = []
        cobertura_knn = []
        fscore_knn = []

        txerro_parzen = []
        precisao_parzen = []
        cobertura_parzen = []
        fscore_parzen = []
        
        txerro_reglog = []
        precisao_reglog = []
        cobertura_reglog = []
        fscore_reglog = []

        txerro_combinacao = []
        precisao_combinacao = []
        cobertura_combinacao = []
        fscore_combinacao = []

################################################################################

# COLETA DAS MÉTRICAS DOS MODELOS TREINADOS

grau_confianca = 0.95
grau_liberdade = n_repeats-1

# Taxa de Erro:
df_txerro = pd.DataFrame(dict_txerro)
medias_txerro = np.array(df_txerro.mean())
sem_txerro = sem(df_txerro.values)
intervalo_txerro = [
    t.interval(grau_confianca, grau_liberdade, 
    loc=medias_txerro[modelo], scale=sem_txerro[modelo])
    for modelo in range(5)
]

# Precisão:
df_precisao = pd.DataFrame(dict_precisao)
medias_precisao = np.array(df_precisao.mean())
sem_precisao = sem(df_precisao.values)
intervalo_precisao = [
    t.interval(grau_confianca, grau_liberdade, 
    loc=medias_precisao[modelo], scale=sem_precisao[modelo])
    for modelo in range(5)
]

# Cobertura:
df_cobertura = pd.DataFrame(dict_cobertura)
medias_cobertura = np.array(df_cobertura.mean())
sem_cobertura = sem(df_cobertura.values)
intervalo_cobertura = [
    t.interval(grau_confianca, grau_liberdade, 
    loc=medias_cobertura[modelo], scale=sem_cobertura[modelo])
    for modelo in range(5)
]

# F-Score:
df_fscore = pd.DataFrame(dict_fscore)
medias_fscore = np.array(df_fscore.mean())
sem_fscore = sem(df_fscore.values)
intervalo_fscore = [
    t.interval(grau_confianca, grau_liberdade, 
    loc=medias_fscore[modelo], scale=sem_fscore[modelo])
    for modelo in range(5)
]

# Montando as matrizes com Valor Pontual e Intervalo de Confiança
matriz_saida = np.concatenate([
    medias_txerro.reshape(-1,1), intervalo_txerro,
    medias_precisao.reshape(-1,1), intervalo_precisao,
    medias_cobertura.reshape(-1,1), intervalo_cobertura,
    medias_fscore.reshape(-1,1), intervalo_fscore,
], axis=1)
np.savetxt(f"ValorPontual_Intervalos.txt", matriz_saida)

# Teste de Friedman:
df_friedman = 1-df_txerro.values
df_nemenyi = 1-df_txerro
stat, p = friedmanchisquare(df_friedman[:,0],df_friedman[:,1],df_friedman[:,2],df_friedman[:,3],df_friedman[:,4])
print(f"Sobre Friedman: stat [{stat}], p_value [{p}]")
if p < 0.05:
    print("Hipótese Nula rejeitada")
    nemenye = nemenyi_teste(df_nemenyi, df_nemenyi.columns)
else:
    print("Hipótese Nula aceita")


nemenye.to_csv("Nemenyi.csv")
