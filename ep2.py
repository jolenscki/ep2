################################
######## NUMERICO - EP2 ########
################################
# Joao Rodrigo Windisch Olenscki
# NUSP 10773224
# Luca Rodrigues Miguel
# NUSP 10705655

# Bibliotecas
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import random
import os
import sys
import time
import datetime
import math
# Typing -> para facilitar documentacao do codigo
from typing import *

# Parametros esteticos do matplotlib
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# Constantes

DPI = 150 # qualidade da imagem de output

def create_folder(folder_list: list, path: str = os.getcwd()) -> str:
    '''
    funcao que cria pastas em um certo diretório
    @parameters:
    - folder_list: cada elemento e uma string contendo o nome da pasta a ser criada
    - path: caminho absoluto anterior as pastas a serem criadas
    -- default_value: os.getcwd(), caminho até a pasta onde o programa está rodando
    @output:
    - new_dir: caminho ate o ultimo diretorio criado
    '''
    for folder_name in folder_list:
        new_dir = os.path.join(path, str(folder_name))
        try:
            os.mkdir(new_dir)
            print("Folder", "{}".format(folder_name), "criado")
        except FileExistsError:
            print("Folder {} already exists in this directory".format(folder_name))
        except TypeError:
            print("TypeError, path = {}, folder_name = {}".format(path, folder_name))
    return new_dir


def open_test_file(filename: str = 'test.txt') -> Tuple[list, np.ndarray]:
    '''
    funcao que abre o arquivo test.txt e extrai dele a lista de posi-
    coes das fontes e os 2049 valores de u_t para a malha especifica 
    onde N = 2048
    @parameters:
    - filename: caminho até o arquivo .txt
    -- default: 'test.txt', considerando que o arquivo esta na mesma
                pasta que o codigo
    @output:
    - p_list: lista das posicoes das fontes
    - u_t: dim = (N+1,), vetor contendo o valor de u_t
    '''
    with open('test.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    p_list = [ float(x) for x in content[0].split('       ') ]
    u_t = [float(x) for x in content[1:]]
    return p_list, u_t

def plot_solution(
                  N: int, output_dic: Dict, test_name: str = 'c', 
                  path: str = os.getcwd(), save: bool = True) -> None:
    '''
    funcao que plota as solucoes exata (retirada do arquivo .txt) e 
    aproximada (calculada pelo metodo dos minimos quadrados) em um mes-
    mo plot
    @parameters:
    - N: numero de divisoes feitas na barra
    - output_dic: dicionario que contem os valores de
    - test_name: teste a ser plotado
    -- default: 'c'
    - path: caminho onde o arquivo sera salvo
    -- default: os.getcwd() (diretorio onde o programa esta rodando)
    - save: flag que indica se o grafico sera salvo ou nao
    -- default: True
    @output:
    - None
    '''
    output_N = output_dic[str(N)]
    a_array = output_N[0]
    u_T = output_N[2]
    u_vectors = output_N[3]
    nf = a_array.shape[0]
    N = u_T.shape[0] + 1

    approx = np.zeros(N-1)
    for k in range(nf):
        approx += a_array[k]*u_vectors[k]
    
    space_array = get_space_array(N)[1:-1]
    
    plt.plot(space_array, u_T, label = r"$u_{T}$", linewidth=4, alpha = 0.5)
    plt.plot(space_array, approx, label = r"$\sum _{i = 1}^{nf}a_{k} u_{k}$",
             linestyle='dashdot', linewidth=1, alpha = 2, color = 'darkred',)
    title_string = r'Gráfico das soluções real e aproximada para $N={}$'.format(N)
    subtitle_string = r'Teste ${}$, T = 1'.format(test_name)
    
    ax = plt.gca()
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'Posição na barra ($x$)')
    ax.set_ylabel(r'Temperatura')
    plt.grid()
    plt.legend()
    if save:
        savedir = os.path.join(path, 'solucao_{}_{}.png'.format(test_name, N))
        plt.savefig(savedir, dpi = DPI, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_punctual_source(output_dic: Dict, delta_t: float, test_name: str = 'c',
                         path: str = os.getcwd(), save: bool = True) -> None:
    '''
    funcao que plota a evolucao das intensidades das fontes conforme e`
    aumentado o valor de N
    @parameters:
    - output_dic: 
    - delta_t: tempo dispendido para rodar o teste em especifico
    - test_name: teste a ser plotado
    -- default: 'c'
    - path: caminho onde o arquivo sera salvo
    -- default: os.getcwd() (diretorio onde o programa esta rodando)
    - save: flag que indica se o grafico sera salvo ou nao
    -- default: True
    @output:
    - None
    '''
    r = [[] for i in range(10)]
    k_list = []
    for k in output_dic.keys():
        k_list.append(int(k))
        for i in range(10):
            r[i].append(output_dic[k][0][i])

    for i in range(10):
        plt.scatter(k_list, r[i], label = r"$a_{%d}$" %(i+1))
        plt.plot(k_list, r[i], 'tab:gray', linestyle='dashdot', linewidth=2, alpha = 0.5)
        plt.annotate('%0.2f' % r[i][-1], xy=(1, r[i][-1]), xytext=(4, 2 * (-1)**i), 
                     xycoords=('axes fraction', 'data'), textcoords='offset points')
    ax = plt.gca()
    ax.set_xscale('log')

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    title_string = r'Intensidade das fontes pontuais para cada valor de $N$'
    subtitle_string = r'Teste ${}$, $T = 1$, tempo de execução = ${:.2f}$ segundos'.format(test_name, delta_t)
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'Intensidade das fontes')
    plt.xticks(k_list)
    plt.grid()
    plt.legend(loc='right', bbox_to_anchor=(1.2, 0.5))
    if save:
        savedir = os.path.join(path, 'intensidade_{}.png'.format(test_name))
        plt.savefig(savedir, dpi = DPI, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
def plot_quadratic_error(output_dic: Dict, eps = float, test_name: str = 'c',
                         path: str = os.getcwd(), save: bool = True) -> None:
    '''
    funcao que plota os valores de E_2 (erro quadratico) para cada N em
    determinado teste
    @parameters:
    - output_dic: 
    - test_name: teste a ser plotado
    -- default: 'c'
    - path: caminho onde o arquivo sera salvo
    -- default: os.getcwd() (diretorio onde o programa esta rodando)
    - save: flag que indica se o grafico sera salvo ou nao
    -- default: True
    @output:
    - None
    '''
    E_2 = []
    k_list = []
    for i, (k, v) in enumerate(output_dic.items()):
        k_list.append(int(k))
        E_2.append(output_dic[k][1])
        plt.scatter(k_list, E_2)
        ax = plt.gca()
        ax.annotate('{:.2E}'.format(E_2[i]), (k_list[i] * 1.03, E_2[i] +.001))
    maxE_2 = max(E_2)

    ax.set_ylim(bottom = 0, top = 1.05*maxE_2)
    ax.set_xscale('log')
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(k_list)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    title_string = r'Valor do erro quadrático em função de $N$'
    subtitle_string = r'Teste ${}$, $T=1$, $\epsilon = {}$'.format(test_name, str(eps))

    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$E_{2}$')
    if save:
        savedir = os.path.join(path, 'erro_quadratico_{}.png'.format(test_name))
        plt.savefig(savedir, dpi = DPI, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def get_M_parameter(T: float, N: int) -> int:
    '''
    funcao para calcular o parametro M
    @parameters:
    - T: constante de tempo T
    - N: numero de divisoes feitas na barra
    @output:
    - M: numero de divisoes no tempo
    '''
    M = T*N
    return int(M)

def get_time_array(M: int, T: float) -> np.ndarray:
    '''
    funcao para criar uma array do tempo, sera usada de forma auxiliar
    para aplicacao de outras funcoes
    @parameters:
    - M: numero de divisoes no tempo
    - T: constante de tempo T
    @return:
    - time_array: dim = (1x(M+1)), contem todos os instantes de tempo
    -- example: [0*(T/M), 1*(T/M), ... , (M-1)*(T/M), M*(T/M)]
    '''
    time_array = np.linspace(0, T, num = M+1)
    return time_array

def get_space_array(N: int) -> np.ndarray:
    '''
    funcao para criar uma array do espaco, sera usada de forma auxiliar
    para aplicacao de outras funcoes
    @parameters:
    - N:  numero de divisoes na barra
    @return:
    - space_array: dim = (1x(N+1)), contem todas as posicoes da barra
    -- example: [0*(1/N), 1*(1/N), ... , (N-1)*(1/N), N*(1/N)]
    '''
    space_array = np.linspace(0, 1, num = N+1)
    return space_array

def add_initial_ending_zeros(array: np.ndarray) -> np.ndarray:
    '''
    funcao para adicionar zeros nos extremos de uma array unidimensio-
    nal, usada para computar a funcao f na eq 11
    @parameters:
    - array: dim = dim, uma array generica
    @output:
    - final_array: dim = dim + 2, a array de entrada com 2 itens adi-
                   cionados: inicial e final, ambos 0
    '''
    final_array = np.zeros(1)
    final_array = np.concatenate((final_array, array))
    final_array = np.concatenate((final_array, np.zeros(1)))
    return final_array

def get_A_matrix(N: int) -> np.matrix:
    '''
    funcao para obter a matriz tri-diagonal A a partir do valor de lambda
    @parameters:
    - N: numero de divisoes na barra
    @output:
    - A_matrix: dim = (N-1)x(N-1)), matriz tridiagonal para o metodo de
                euler implicito e para o metodo de crank-nicolson
    '''
    a = np.diagflat(- N/2 * np.ones(N-2), 1)
    b = np.diagflat((1+N) * np.ones(N-1))
    c = np.diagflat(- N/2 * np.ones(N-2), -1)
    A_matrix = np.matrix(a+b+c)
    return A_matrix

def perform_ldlt_transformation_sparse(
            a: np.ndarray,
            b: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    funcao que aplica a decomposicao de uma matriz tridiagonal simetrica
    na multiplicacao de 3 outras, de forma que A = L.D.L'
    @parameters:
    - a: array unidimensional generica que representa os valores da 
         diagonal principal
    - b: array unidimensional generica que representa os valores da 
         diagonal secundaria, seu primeiro valor e nulo (b[0] = 0)
    -> a e b possuem a mesma dimensao
    @output:
    - d: array unidimensional de mesma dimensao que a e b e que repre-
         senta a diagonal principal da matriz D
    - l: array unidimensional de mesma dimensao que a e b e que repre-
         senta a diagonal secundaria inferior da matriz L e também a diagonal 
         secundaria superior de L', seu primeiro valor e nulo (l[0] = 0)
    '''
    N = a.shape[0] + 1
    d = np.ones(N-1)
    l = np.ones(N-1)
    d[0] = a[0]
    for i in range(N-2):
        l[i+1] = b[i+1]/d[i]
        d[i+1] = a[i+1] - b[i+1]*l[i+1]
    return d, l

def solve_linear_system_sparse(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    '''
    funcao que resolve um sistema linear da forma Ax = u para A matriz
    trigonal simétrica esparsa, retornando o vetor x
    @parameters:
    - A: dim = ((N-1)x(N-1)), matriz quadrada de entrada do sistema
    - u: dim = ((N-1)x1), vetor de saida da equacao
    @output:
    - x: dim = ((N-1)x1), vetor de incognitas
    '''
    # A.x = L.D.Lt.x
    a = np.asarray(A.diagonal(0)).ravel()
    N = a.shape[0] + 1
    b = np.concatenate((np.array([0]), np.asarray(A.diagonal(1)).ravel()))
    d, l = perform_ldlt_transformation_sparse(a, b)
    
    # Como explicado no relatorio, resolvemos o problema atraves da reso-
    # lucao de 3 loops: primeiro resolvemos em z, depois em y e ai sim em
    # x
    #Loop em z:
    z = np.zeros(N-1)
    z[0] =  u[0]
    for i in range(1, N-1):
        z[i] = u[i] - z[i-1]*l[i]
    #Loop em y:
    y = np.zeros(N-1)
    for i in range(N-1):
        y[i] = z[i]/d[i]
    #Loop em x:
    x = np.zeros(N-1)
    x[-1] = y[-1]
    for i in range(N-3, -1, -1):
        x[i] = y[i] - x[i+1]*l[i+1]
    return x

def perform_ldlt_transformation(
            A: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    funcao que aplica a decomposicao de uma matriz A = L.D.L'
    @parameters:
    - A: dim = (nf x nf), matriz quadrada e simetrica a ser decomposta
    @output:
    - L: dim = (nf x nf), matriz quadrada com valores na parte inferior
    - D: dim = (nf x nf), matriz quadrada com valores na diagonal
    '''
    nf = A.shape[0]
    L = np.eye(nf)
    D = np.zeros([nf, nf])
    for i in range(nf):
        D[i, i] = A[i, i]
        for j in range(i):
            L[i, j] = A[i, j]
            for k in range(j):
                L[i, j] -= L[i, k]*L[j, k]*D[k, k]
            
            L[i, j] /= D[j, j]
            D[i, i] -= L[i, j]*L[i, j]*D[j, j]
    
    return L, D

def solve_linear_system(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    '''
    funcao que resolve o sistema linear do tipo Ax = u para uma matriz
    simetrica nao-esparsa, retornando o vetor x
    @parameters:
    - A: dim = (nf x nf), matriz quadrada do sistema normal
    - u: dim = (nf x 1), matriz de resultados do sistema normal
    @output:
    - x: dim = (nf x 1), matriz das incognitas do sistema normal
    '''
    nf = u.shape[0]
    L, D = perform_ldlt_transformation(A)
    
    z = np.zeros(nf)
    for i in range(nf):
        bckwrd_sum = 0
        for k in range(i):
            bckwrd_sum += L[i, k]*z[k]
        z[i] = u[i] - bckwrd_sum
    
    y = np.zeros(nf)
    for i in range(nf):
        y[i] = z[i]/D[i,i]

    x = np.zeros(nf)
    for i in range(nf-1, -1, -1):
        bckwrd_sum = 0
        for k in range(i+1, nf):
            bckwrd_sum += L[k, i]*x[k]
        x[i] = y[i] - bckwrd_sum

    return x

def f(
            space_array: np.ndarray, 
            k: int,
            T: float,
            M: int,
            p: float
            ) -> np.ndarray:
    '''
    funcao piecewise (degrau) para modelamento das fontes pontuais
    @parameters:
    - space_array: dim = (1x(N+1)), contem todas as posicoes da barra
    - k: indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: constante de tempo T
    - M: numero de divisoes no tempo
    - p: 0 < p < 1, posicao da fonte de calor
    @output:
    - f_array: dim = (N+1,), contem os valores de f calculados para um
               instante especifico k para as posicoes da barra exceto 
               as extremas, estas sao substituidas por zeros
    '''
    N = space_array.shape[0] - 1
    h = 1/N
    t_delta = 1e-10 # Pequeno delta a ser adicionado aos intervalos de forma a
                    # alarga-los e permitir que os erros de truncamento do python
                    # e do np sejam equiparados.
    t = k*T/M
    x = space_array
    inf = p - h/2 - t_delta
    sup = p + h/2 + t_delta
    g_array = np.piecewise(x,
                              [x < inf, (x >= p - h/2) & (x <= p + h/2), x > sup],
                              [   0   ,               1/h              ,   0    ]
                             )
    r = 10*(1 + np.cos(5*t))
    f_array = g_array*r
    f_array = add_initial_ending_zeros(f_array[1:-1])
    return f_array

def crank_nicolson(
            T: float, 
            M: int, 
            u: np.ndarray, 
            space_array: np.ndarray, 
            p: float
            ) -> np.ndarray:
    '''
    funcao iterativa que realiza as integracoes numericas pelo metodo 
    de crank-nicolson
    @parameters:
    - T: constante de tempo T
    - M: numero de divisoes no tempo
    - u: dim = ((N+1)x(N+1)), matriz de temperaturas, inicialmente nulas
    - space_array: dim = (1x(N+1)), contem todas as posicoes da barra
    - p: posicao da fonte pontual
    @output:
    - u: dim = ((M+1)x(N+1)), matriz de temperaturas com seus valores
         calculados em T = 1
    '''
    N = u.shape[0] - 1
    A = get_A_matrix(N)
    dt = T/M
    
    f_array_anterior = f(space_array, 0, T, M, p) # primeiro valor de f
    for k in range(1, N+1):
        f_array_atual = f(space_array, k, T, M, p)
        f_mean = (dt/2)*(f_array_anterior + f_array_atual)
        f_array_anterior = f_array_atual
        upper_element = np.array([
                                    (1 - N)*u[1] +
                                    (N/2)*u[2] +
                                    f_mean[1]
                                ])
        mid_elements = np.asarray([
                                    (1 - N)*u[2:N-1] +
                                    (N/2) * (u[1:N-2] + u[3:N]) +
                                    f_mean[2:N-1]
                                ]).ravel()
        lower_element = np.array([
                                    (1 - N)*u[N-1] +
                                    (N/2)*u[N-2] +
                                    f_mean[N-1]
                                ])
        linsys_asw = np.concatenate((upper_element, mid_elements, lower_element))
        u[1:N] = solve_linear_system_sparse(A, linsys_asw)
    return u

def get_u_p(N: int, T: int, p: float) -> np.ndarray:
    '''
    funcao que calcula a matriz de temperaturas u_p para uma fonte pon-
    tual em x = p utilizando o metodo de crank-nicolson, retornando 
    entao a matriz de temperaturas em T = 1 para esta forcante
    @parameters:
    - N: numero de divisoes na barra
    - T: constante de tempo T
    - p: posicao da fonte pontual
    @output:
    - u: dim = (N-1,), matriz de temperaturas em T = 1
    '''
    M = get_M_parameter(T, N)
    time_array = get_time_array(M, T)
    space_array = get_space_array(N)
    
    u = np.zeros(N+1) # condicao de contorno, u(t = 0) = 0
    u = crank_nicolson(T, M, u, space_array, p)
    u = u[1:-1]
    
    return u

def generate_unf_vectors(N: int, T: int, p_list: list) -> list:
    '''
    funcao que gera uma lista de vetores u_1,...,u_nf referentes as 
    temperaturas causadas pelas forcantes p_1,..., p_nf em T = 1
    @parameters:
    - N: numero de divisoes na barra
    - T: constante de tempo T
    - p_list: lista de posicoes p_1,...,p_nf onde estao as forcantes
    @output:
    - u_vectors: lista de vetores (arrays) u_p
    '''
    nf = len(p_list)
    u_vectors = []
    for i in range(nf):
        ui = get_u_p(N, T, p_list[i])
        u_vectors.append(ui)
    return u_vectors

def create_normal_system(
            u_vectors: list, 
            u_T: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    funcao que cria o sistma normal, isto e`, dada uma lista de vetores
    u (u_1,...,u_nf) e o vetor u_T que representa o comportamento da so-
    lucao da equacao de calor gera a matriz quadrada simetrica A dos 
    produtos internos entre u_i e u_j (i, j = 1,..., nf) e a matriz dos
    resultados b, que contem os produtos internos entre u_i e u_T (i =
    1,...,nf)
    @parameters:
    - u_vectors: lista de vetores (arrays) u_p
    - u_T: dim = (N+1,), array da solucao obtida pela solucao da equa-
           cao de calor
    @output:
    - A: matriz quadrada do sistema normal
    - b: matriz dos resultados do sistema normal
    '''
    nf = len(u_vectors)
    A = np.zeros([nf, nf])
    b = np.zeros(nf)
    for i in range(nf):
        b[i] = np.inner(u_vectors[i], u_T)
        for j in range(nf):
            inner_product = np.inner(u_vectors[i], u_vectors[j])
            A[i][j] = inner_product
            A[j][i] = inner_product
    
    return A, b

def calculate_quadratic_error(
            N: int, 
            u_T: np.ndarray, 
            a_list: list,
            u_vectors: list
            ) -> float:
    '''
    funcao que calcula o erro quadratico
    @parameters:
    - N: numero de divisoes da barra
    - u_T: dim = (N+1,), array da solucao obtida pela solucao da equa-
           cao de calor
    - a_list: lista das intensidades das forcantes
    - u_vectors: lista de vetores (arrays) u_p
    @output:
    - E_2: erro quadratico relacionado as resolucao do sistema normal
    '''
    dx = 1/N
    a_array = np.asarray(a_list)
    nf = a_array.shape[0]

    approx = np.zeros(N-1)
    for k in range(nf):
        approx += a_array[k]*u_vectors[k]

    error = u_T - approx
    quadratic_error = 0
    for i in range(N-1):
        quadratic_error += error[i]**2

    E_2 = np.sqrt(dx * quadratic_error)
    return E_2


def run_test(test_name: str, N: int, eps: float = 1e-2) -> Union[np.ndarray, Tuple[Dict, float]]:
    '''
    funcao auxiliar que roda os testes
    @parameters:
    - test_name: teste a ser executado
    - N: numero de divisoes na barra
    - eps: ruido no teste d
    @output:
    -~ x: array com as intensidades das fontes pontuais
    -~ output_dic: dicionario que contem
       * x: array com as intensidades das fontes pontuais
       * E_2: erro quadratico
       * u_T: array u_T da solucao da equacao do calor
       * u_vectors: lista de vetores (arrays) u_p
    -~ delta_t: tempo decorrido para a execucao do teste
    '''
    
    def test_a(N: int, T: float, p_list: list) -> np.ndarray:
        u_vectors = generate_unf_vectors(N, T, p_list)
        u_T = 7*u_vectors[0]

        A, b = create_normal_system(u_vectors, u_T)
        x = solve_linear_system(A, b)

        return x

    def test_b(N: int, T: float, p_list: list) -> np.ndarray:
        u_vectors = generate_unf_vectors(N,  T, p_list)
        u_T = 2.3*u_vectors[0] + 3.7*u_vectors[1] \
              + 0.3*u_vectors[2] + 4.2*u_vectors[3]

        A, b = create_normal_system(u_vectors, u_T)
        x = solve_linear_system(A, b)

        return x

    def test_c(N_list: list, T: float) -> Tuple[Dict, float]:
        start = time.time()
        output_dict = {}
        for N in N_list:
            print(f"N = {N}")
            it = int(2048/N)

            p_list, u_T = open_test_file(filename = 'test.txt')
            u_T = u_T[::it][1:-1]
            u_T = np.asarray(u_T)

            u_vectors = generate_unf_vectors(N,  T, p_list)

            A, b = create_normal_system(u_vectors, u_T)

            x = solve_linear_system(A, b)
            for i in range(len(x)):
                print(f"     a{i+1} = {x[i]}")
            E_2 = calculate_quadratic_error(N, u_T, x, u_vectors)
            print(f"     E_2 = {E_2}")
            output_dict[str(N)] = [x, E_2, u_T, u_vectors]
        delta_t = time.time() - start
        return output_dict, delta_t

    def test_d(
                N_list: list, 
                T: float, 
                eps: float
                ) -> Tuple[Dict, float]:

        start = time.time()
        output_dict = {}
        for N in N_list:
            print(f"N = {N}")
            it = int(2048/N)

            p_list, u_T = open_test_file(filename = 'test.txt')
            u_T = u_T[::it][1:-1]
            u_T = np.asarray(u_T)

            for i in range(N-1):
                r = (random.random() - 0.5)*2
                m = 1 + r*eps
                u_T[i] = m * u_T[i]

            u_vectors = generate_unf_vectors(N,  T, p_list)

            A, b = create_normal_system(u_vectors, u_T)

            x = solve_linear_system(A, b)
            for i in range(len(x)):
                print(f"     a{i+1} = {x[i]}")
            E_2 = calculate_quadratic_error(N, u_T, x, u_vectors)
            print(f"     E_2 = {E_2}")
            output_dict[str(N)] = [x, E_2, u_T, u_vectors]
        delta_t = time.time() - start
        return output_dict, delta_t
    
    T = 1
    N_list = [128, 256, 512, 1024, 2048]
    
    if test_name == 'a':
        N = N_list[0]
        p_list = [0.35]
        x = test_a(N, T, p_list)
        for i in range(len(x)):
            print(f'a{i+1} = {x[i]}')
        return x
        
    elif test_name == 'b':
        N = N_list[0]
        p_list = [0.15, 0.3, 0.7, 0.8]
        x = test_b(N, T, p_list)
        for i in range(len(x)):
            print(f'a{i+1} = {x[i]}')
        return x
        
    elif test_name == 'c':
        output_dic, delta_t = test_c([N], T)
        return  output_dic, delta_t
        
    elif test_name == 'd':
        output_dic, delta_t = test_d([N], T, eps)
        return output_dic, delta_t
    
    elif test_name == 'cN_list':
        output_dic, delta_t = test_c(N_list, T)
        return output_dic, delta_t
    
    elif test_name == 'dN_list':
        output_dic, delta_t = test_d(N_list, T, eps)
        return output_dic, delta_t
        
    else:
        raise KeyError(f"'{test_name}' não é uma opção válida!\
                       As opções são 'a', 'b', 'c', 'd'")
    
def run_plots() -> None:
    '''
    funcao que gera todos os plots para todos os N's de ambos os testes
    @parameters:
    - outputs: dicionario que guarda os resultados dos testes 'c' e 'd'
    @output
    - None
    '''
    N_list = [128, 256, 512, 1024, 2048]
    T = 1
    d_output, delta_t_d = run_test('dN_list', 128)
    c_output, delta_t_c = run_test('cN_list', 128)
    outputs = {
             'd': [d_output, delta_t_d],
             'c': [c_output, delta_t_c]}
    path = create_folder(['figures'], path = os.getcwd())
    for k, v in outputs.items():
        output_dic = outputs[k][0]
        delta_t = outputs[k][1]
        for N in N_list:
            plot_solution(N, output_dic, test_name = k, path = path)
        plot_punctual_source(output_dic, delta_t, test_name = k, path = path)
        plot_quadratic_error(output_dic, 1e-2, test_name = k, path = path)    

def main() -> None:
    '''
    funcao main que executa o programa, requere os testes e Ns do usua-
    rio e retorna as intensidades das fontes pontuais (e o erro quadra-
    tico, se aplicavel)
    @parameters:
    - None
    @output:
    - None
    '''
    test_name = input("Digite o teste que quer rodar \n (opções: 'a', 'b', 'c', 'd'):")
    if test_name == 'c' or test_name == 'd':
        N = int(input("Digite o valor de N para o teste \n (opções = 128, 256, 512, 1024, 2048):"))
    else:
        N = 128
    run_test(test_name, N)

if __name__ == '__main__':
    main()