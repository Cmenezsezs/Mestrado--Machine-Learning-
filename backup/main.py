import numpy as np
import random
from base_dados.base import carrega_base, normaliza_0_1
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, f1_score, confusion_matrix, accuracy_score


# Algoritmo 1: Protótipos
def algo_1(atrib, Uik, C, V):
    prototipos = np.zeros([C, V])
    for k in range(C):

        # 1: Rank
        for j in range(V):
            ziped = zip(Uik[:, k], atrib[:, j])
            ord_j = np.asarray(sorted(ziped, key=lambda x: x[1]))

            # 2, 3:
            soma = - np.sum(np.abs(ord_j[:, 0]))
            count = 0
            soma_anterior = soma
            while(soma < 0):
                soma += 2 * np.abs(ord_j[count, 0])
                soma_anterior = soma
                count += 1

            # 5
            if (soma_anterior == 0 and soma == 0):
                # 6
                prototipos[k, j] = (ord_j[count, 1] + ord_j[count-1, 1])/2
            else:
                prototipos[k, j] = ord_j[count-1, 1]  # 4
    return prototipos


# Peso da relevância das variáveis
def equa_26(atrib, Uik, prototipos, Tv, C, V, P):
    Vkj = np.zeros([C, V])
    for k in range(C):
        for j in range(V):
            numerador = 0.0
            for i in range(P):
                numerador += Uik[i, k]*np.abs(atrib[i, j]-prototipos[k, j])
            numerador = np.exp(-numerador/Tv)

            denominador = 0.0
            for w in range(V):
                aux = 0.0
                for i in range(P):
                    aux += Uik[i, k]*np.abs(atrib[i, w]-prototipos[k, w])
                denominador += np.exp(-aux/Tv)
            Vkj[k, j] = numerador/denominador
    return Vkj


# Calculo do Grau de Pertinência
def equa_27(atrib, peso_relevancia, prototipos, Tu, C, V, P):
    Uik = np.zeros([P, C])
    for k in range(C):
        for i in range(P):
            numerador = 0.0
            for j in range(V):
                numerador += peso_relevancia[k, j] * \
                    np.abs(atrib[i, j] - prototipos[k, j])
            numerador = np.exp(-numerador/Tu)

            denominador = 0.0
            for w in range(C):
                aux = 0.0
                for j in range(V):
                    aux += peso_relevancia[w, j] * \
                        np.abs(atrib[i, j] - prototipos[w, j])
                denominador += np.exp(-aux/Tu)
            Uik[i, k] = numerador/denominador
    return Uik


# J_EFCM-LS1
def funcao_obj(atrib, Uik, prototipos, peso_relevancia, Tu, Tv, C, P, V):

    # Parte 1
    escalar_p1 = 0.0
    for k in range(C):
        for i in range(P):
            aux = 0.0
            for j in range(V):
                aux += peso_relevancia[k, j] * \
                    np.abs(atrib[i, j] - prototipos[k, j])
            escalar_p1 += Uik[i, k] * aux

    # Parte 2
    escalar_p2 = 0.0
    for k in range(C):
        for i in range(P):
            escalar_p2 += Uik[i, k] * np.log(Uik[i, k])

    # Parte 3
    escalar_p3 = 0.0
    for k in range(C):
        for j in range(V):
            escalar_p3 += peso_relevancia[k, j] * np.log(peso_relevancia[k, j])

    return escalar_p1 + (Tu*escalar_p2) + (Tv*escalar_p3)


def algo_2(atrib, labels, C, T, Tu, Tv, epsilon):
    P = len(labels)  # Número de amostras (num de objs)
    V = len(atrib[0])  # Número de Labels/Features por objeto

    # 1: Initialiation
    Uik = np.random.rand(P, C)
    for i in range(P):
        Uik[i] /= np.sum(Uik[i])  # Soma de cada linha dá 1
    Uikt1 = np.random.rand(P, C)

    # 2: Repeat
    tzinho = 0
    while (tzinho < T and np.max(np.abs(Uik - Uikt1)) >= epsilon):
        Uikt1 = Uik.copy()

        # Step 1
        prototipos = algo_1(atrib, Uik, C, V)

        # Step 2: Peso da relevância das variáveis
        peso_relevancia = equa_26(atrib, Uik, prototipos, Tv, C, V, P)

        # Step 3: Calculo do Grau de Pertinência
        Uik = equa_27(atrib, peso_relevancia, prototipos, Tu, C, V, P)
        tzinho += 1

    return Uik, tzinho, peso_relevancia, prototipos


def modified_part_coeficient(Uik, P, C):
    Pc = ((Uik**2).sum())/P
    return 1 - (C/(C-1))*(1-Pc)


# TODO: Escolher um log e indicar no relatório (a gente quem escolhes)
def partition_entropy(Uik, P):
    M = -(Uik * np.log(Uik))
    M = np.where(np.isnan(M), 0, M)  # para números muito pequenos
    return M.sum()/P


def crisp_partition(Uik, C, printar=False):
    CP = np.argmax(Uik, axis=1)
    if printar:
        print(f"Count Crisp Partition: ", end="")
        for i in range(C):
            print(f"class_{i+1}:[{np.sum(CP==i)}] ", end="")
        print()
    return CP

# comparar particoes crisp em 7 grupos 2 a 2, com indice de hand corrigido e f measure
# imprimir prototipos do melhor resultado, matrix de confusão e resulto a priori
# matrix de confusao de uma particao crisp vs a outra (comparar as matrizes de confusao)
# esquema do peso e relevancia


if __name__ == '__main__':
    # Dir para base 2 [.cvs] (construída por othon usando pandas)
    dir_treino = "./base_dados/segmentation.data"
    dir_test = "./base_dados/segmentation.test"

    colunas_base_1 = [4, 5, 6, 7, 8, 9]
    colunas_base_2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    colunas_base_3 = colunas_base_1 + colunas_base_2
    bases = {'base1': colunas_base_1,
             'base2': colunas_base_2,
             'base3': colunas_base_3}

    # seed = np.random.randint(1, 9999)
    seed = 0
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    print(f"Seed Usado: {seed}")

    for i_base in range(len(bases)):
        key_base = list(bases.keys())[i_base]
        print(f"Início da execução com base: {key_base}")

        atrib_train, labels_train, escala, encoder = carrega_base(
            dir_treino, bases[key_base], "train", shuffle=True, seed=seed,
            plotar_grafico=False)

        atrib_test, labels_test, _, __ = carrega_base(
            dir_test, bases[key_base], "test", escala=escala, encoder=encoder,
            shuffle=True, seed=seed, plotar_grafico=False)

        ############################################################################

        # Tu_vector = np.linspace(1, 100, 5)
        Tu_vector = np.arange(0.01, 100, 0.1)

        # Tv_vector = np.linspace(10, 10**8, 5)
        # Tv_vector = np.arange(10, 10**8, 10)
        Tv_vector = np.arange(0.01, 250, 1)

        C = 7  # Número de Clusters
        T = 150  # Número de Iterações
        epsilon = 10**-10
        numeradoro_tentativas = 10
        salvar_em_arquivos = False
        usar_acuracia_sklearn = False

        ############################################################################

        malha = []  # Para construção da malha
        relacao_ari_obj = []
        melhor_perda = 0.0
        if usar_acuracia_sklearn:
            melhor_perda = 10**-10
        else:
            melhor_perda = 10**10
        Tu_melhor_perda, Tv_melhor_perda, Uik_salvo = None, None, None
        ite_salvo, peso_relevancia_salvo, CP_salvo, prototipos_salvo = None, None, None, None

        for i in range(numeradoro_tentativas):
            print(f"Testando exec {i+1} da base {key_base}", end="\r")
            Tu = np.random.choice(Tu_vector)
            Tv = np.random.choice(Tv_vector)

            Uik, ite, peso_relevancia, prototipos = algo_2(
                atrib_train, labels_train, C, T, Tu, Tv, epsilon)

            CP = crisp_partition(Uik, C, printar=False)

            perda = 0.0
            if usar_acuracia_sklearn:
                perda = accuracy_score(labels_train, CP)
            else:
                perda = funcao_obj(
                    atrib_train, Uik, prototipos, peso_relevancia, Tu, Tv, C,
                    len(labels_train), len(atrib_train[0]))

            malha.append([Tu, Tv, perda])
            if (usar_acuracia_sklearn and perda > melhor_perda) or (not usar_acuracia_sklearn and perda < melhor_perda):
                Tu_melhor_perda, Tv_melhor_perda, melhor_perda = Tu, Tv, perda
                prototipos_salvo, ite_salvo = prototipos, ite
                peso_relevancia_salvo, CP_salvo = peso_relevancia, CP
                Uik_salvo = Uik

            relacao_ari_obj.append(
                [adjusted_rand_score(labels_train, CP), perda])

        print()
        print(
            f"Melhor Tu: {Tu_melhor_perda}, Tv: {Tv_melhor_perda}, ite: {ite_salvo}")

        # Item 1: Printar Protótipos
        print(f"Protótipos:\n{prototipos_salvo}")
        print(f"Grau de Pertinência Train:\n{Uik_salvo}")
        print("> Perda da base de Train[func_obj_artigo(train)]: ", end="")
        print(funcao_obj(atrib_train, Uik_salvo, prototipos_salvo,
                         peso_relevancia_salvo, Tu_melhor_perda,
                         Tv_melhor_perda, C, len(labels_train),
                         len(atrib_train[0])))
        print("> Perda da base de Train [acuracia(train)]: ", end="")
        print(accuracy_score(labels_train, CP_salvo))
        print("--------------")

        # Item 2: a matrix de confusão da partição crisp versus a partição a priori
        # >> Comparação usando base de test
        Uik_test = equa_27(atrib_test, peso_relevancia_salvo, prototipos_salvo,
                           Tu_melhor_perda, C, len(atrib_test[0]), len(labels_test))
        CP_test = crisp_partition(Uik_test, C, printar=False)

        # Perda da base de este com melhor config encontrada
        perda_test = funcao_obj(
            atrib_test, Uik_test, prototipos_salvo,
            peso_relevancia_salvo, Tu_melhor_perda, Tv_melhor_perda, C,
            len(labels_test), len(atrib_test[0]))
        print(f"> func_obj_artigo(test_dataset): {perda_test}")
        print(f"> acuracia(test): {accuracy_score(labels_test, CP_test)}")
        print("--------------")

        # Item 3:
        # Modified Partition da base de treino de cada dataset usando melhor config
        print("modified_part_coeficient_test: ",
              modified_part_coeficient(Uik_test, len(labels_test), C))
        print(f"ARI_test: {adjusted_rand_score(labels_test, CP_test)}")
        # Compute the F1 score, also known as balanced F-score or F-measure.
        print(
            f"F_Measure_test: {f1_score(labels_test, CP_test, average='weighted')}")
        print("Matriz de Confusão Train:\n",
              confusion_matrix(labels_train, CP_salvo))
        print("Matriz de Confusão Test:\n",
              confusion_matrix(labels_test, CP_test))
        print("--------------", end="\n\n")

        if salvar_em_arquivos:
            # Plot da malha
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            malha = np.asarray(malha)
            ax.plot_trisurf(malha[:, 0], malha[:, 1], malha[:, 2])
            ax.set_title("Malha (Tu, Tv, altura=ver_no_codigo)")
            plt.savefig(f'./salvos/perda_{key_base}.png')
            # plt.show()

            # Plota a relação entre o ARI e a perda
            plt.clf()
            x = [a for a in range(len(relacao_ari_obj))]
            ari_s = np.asarray([float(a[0]) for a in relacao_ari_obj])
            obj_s = np.asarray([float(a[1]) for a in relacao_ari_obj])

            ari_s = normaliza_0_1(ari_s)
            obj_s = normaliza_0_1(obj_s)

            plt.plot(x, ari_s, label="ari")
            plt.plot(x, obj_s, label="obj")
            plt.legend()
            plt.savefig(f'./salvos/ARI_perda_{key_base}.png')
            # plt.show()

            # salva em arquivos
            np.savetxt(f"./salvos/{key_base}_CP_teste.txt",
                       np.asarray(CP_test))
            np.savetxt(f"./salvos/{key_base}_labels_test.txt",
                       np.asarray(labels_test))
            np.savetxt(f"./salvos/{key_base}_prototipos.txt",
                       np.asarray(prototipos_salvo))
            np.savetxt(f"./salvos/{key_base}_peso_relevancia.txt",
                       np.asarray(peso_relevancia_salvo))

    print()
    print("===================================================================")
    print(f"Fim da execução com base: {key_base}")
    print("===================================================================")
    print()
