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
import os
import sys
import time
import datetime
import math

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

def create_folder(folder_list, path = os.getcwd()):
    '''
    funcao que cria pastas em um certo diretório
    @parameters:
    - folder_list: lista, cada elemento e uma string contendo o nome da pasta a ser criada
    - path: string, caminho absoluto anterior as pastas a serem criadas
    -- default_value: os.getcwd(), caminho até a pasta onde o programa está rodando
    @output:
    - None
    '''
    for folder_name in folder_list:
        try:
            os.mkdir(os.path.join(path, str(folder_name)))
            print("Folder", "{}".format(folder_name), "criado")
        except FileExistsError:
            print("Folder {} already exists in this directory".format(folder_name))
        except TypeError:
            print("TypeError, path = {}, folder_name = {}".format(path, folder_name))
            


def open_test_file(filename = 'test.txt'):
    '''
    funcao que abre o arquivo test.txt e extrai dele a lista
    de posicoes das fontes e os 2049 valores de u_t para a malha
    especifica onde N = 2048
    @parameters:
    - filename: string, caminho até o arquivo .txt
    -- default: 'test.txt', considerando que o arquivo esta na mes-
                ma pasta que o codigo
    @output:
    - p_list: list, lista das posicoes das fontes
    - u_t: array (N+1,), vetor contendo o valor de u_t
    '''
    with open('test.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    p_list = [ float(x) for x in content[0].split('       ') ]
    u_t = [float(x) for x in content[1:]]
    return p_list, u_t

def get_M_parameter(T, N):
    '''
    funcao para calcular o parametro M
    @parameters:
    - T: float, constante de tempo T
    - N: inteiro, numero de divisoes feitas na barra
    @output:
    - M: inteiro, numero de divisoes no tempo
    '''
    M = T*N
    return int(M)

def get_time_array(M, T):
    '''
    funcao para criar uma array do tempo, sera usada de forma auxiliar para aplicacao de outras funcoes
    @parameters:
    - M: inteiro, numero de divisoes no tempo
    - T: float, constante de tempo T
    @return:
    - time_array: array (1x(M+1)), contem todos os instantes de tempo
    -- example: [0*(T/M), 1*(T/M), ... , (M-1)*(T/M), M*(T/M)]
    '''
    time_array = np.linspace(0, T, num = M+1)
    return time_array

def get_space_array(N):
    '''
    funcao para criar uma array do espaco, sera usada de forma auxiliar para aplicacao de outras funcoes
    @parameters:
    - N: inteiro, numero de divisoes na barra
    @return:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    -- example: [0*(1/N), 1*(1/N), ... , (N-1)*(1/N), N*(1/N)]
    '''
    space_array = np.linspace(0, 1, num = N+1)
    return space_array

def add_initial_ending_zeros(array):
    '''
    funcao para adicionar zeros nos extremos de uma array unidimensional, usada para computar a funcao f na eq 11
    @parameters:
    - array: array, uma array generica
    @output:
    - final_array: array, a array de entrada com 2 itens adicionados: inicial e final, ambos 0
    '''
    final_array = np.zeros(1)
    final_array = np.concatenate((final_array, array))
    final_array = np.concatenate((final_array, np.zeros(1)))
    return final_array

def get_A_matrix(N):
    '''
    funcao para obter a matriz tri-diagonal A a partir do valor de lambda
    @parameters:
    - N: inteiro, numero de divisoes na barra
    @output:
    - A_matrix: matrix ((N-1)x(N-1)), matriz tridiagonal para o metodo de euler implicito
                e para o metodo de crank-nicolson
    '''
    a = np.diagflat(- N/2 * np.ones(N-2), 1)
    b = np.diagflat((1+N) * np.ones(N-1))
    c = np.diagflat(- N/2 * np.ones(N-2), -1)
    A_matrix = np.matrix(a+b+c)
    return A_matrix

def perform_ldlt_transformation(a, b):
    '''
    funcao que aplica a decomposicao de uma matriz tridiagonal simetrica
    na multiplicacao de 3 outras, de forma que A = L.D.L'
    @parameters:
    - a: array, array unidimensional generica que representa os valores da 
         diagonal principal
    - b: array, array unidimensional generica que representa os valores da 
         diagonal secundaria, seu primeiro valor e nulo (b[0] = 0)
    -> a e b possuem a mesma dimensao
    @output:
    - d: array, array unidimensional de mesma dimensao que a e b e que repre-
         senta a diagonal principal da matriz D
    - l: array, array unidimensional de mesma dimensao que a e b e que repre-
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

def solve_linear_system(A, u):
    '''
    funcao que resolve um sistema linear da forma Ax = u para A matriz
    trigonal simétrica, retornando o vetor x
    @parameters:
    - A: array ((N-1)x(N-1)), matriz quadrada de entrada do sistema
    - u: array ((N-1)x1), vetor de saida da equacao
    @output:
    - x: array ((N-1)x1), vetor de incognitas
    '''
    # A.x = L.D.Lt.x
    a = np.asarray(A.diagonal(0)).ravel()
    N = a.shape[0] + 1
    b = np.concatenate((np.array([0]), np.asarray(A.diagonal(1)).ravel()))
    d, l = perform_ldlt_transformation(a, b)
    
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

def f(space_array, k, T, M, p):
    '''
    funcao que define a f(x,t) para o item c) da primeira tarefa
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    - p: float, 0 < p < 1 posicao da fonte de calor
    @output:
    - f_array: array (N+1,), contem os valores de f calculados para um instante especifico k 
                para as posicoes da barra exceto as extremas, estas sao substituidas por zeros
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

def crank_nicolson(T, M, u, space_array, p):
    ''''''
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
        u[1:N] = solve_linear_system(A, linsys_asw)
    return u

def get_u_p(N, T, p):
    ''''''
    M = get_M_parameter(T, N)
    time_array = get_time_array(M, T)
    space_array = get_space_array(N)
    
    u = np.zeros(N+1) # condicao de contorno, u(t = 0) = 0
    u = crank_nicolson(T, M, u, space_array, p)
    
    return u

def generate_unf_vectors(N, T, p_list):
    nf = len(p_list)
    u_vectors = []
    for i in range(nf):
        ui = get_u_p(N, T, p_list[i])
        u_vectors.append(ui)
    return u_vectors