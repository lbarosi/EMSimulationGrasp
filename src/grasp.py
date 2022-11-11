# -*- coding: utf-8 -*-
"""Este módulo provê ferramentas para automação do uso do GRASP."""
import glob
import os
import multiprocessing
import shutil
import shlex
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#######################################
# Funções para manipulação de cuts
#######################################


def read_cut(filename):
    """Lê o arquivo cut indicado no argumento e retorna dataframe com componentes reais e complexas das polarizações dos campos."""
    # Lê primeiras duas linhas, descarta primeira e cria lista float
    with open(filename) as file:
        head = list(map(float,
                        [next(file) for x in range(2)][1].strip().split()))
    # Parsing do cabeçalho
    thetas = head[0] + head[1] * np.arange(head[2])  # ângulos
    data = pd.read_table(filename, skiprows=2, header=None, sep="\s+")
    # tiding
    data.columns = ["Eco", "ImEco", "Ecx", "ImEcx"]
    data["theta"] = thetas
    return data

def load_cuts(mask):
    cut_files = glob.glob(mask, recursive=True)
    dfs = []
    gains = []
    gains_max = []
    params = []
    for file in cut_files:
        df = read_cut(file)
        dfs.append(df)
        params.append(float(file.split("/")[-2].split("_")[-1]))
        gains.append(gain_dB(df))
        gains_max.append(gain_max(df))
    df_gains = pd.DataFrame({"Values": params, "Gain": gains,
                            "Gain_Max": gains_max})
    return df_gains, dfs

def plot_beam_pattern(data, ax=None, label="padrão", norm=True):
    """Gráfico de padrão de feixe normalizado a partir de dataframe.

    Args:
        data (type): dataframe tipo cut.
        ax (type): matplotlib `ax`. Defaults to None.
        label (type): rótulo dos dados `label`. Defaults to "padrão".
        norm (bool): booleano para normalizar os dados ou não pela amplitude
                     máxima.

    Returns:
        type: matplotlib ax

    """
    EE = data.Eco ** 2 + data.ImEco ** 2
    EEx = data.Ecx ** 2 + data.ImEcx ** 2
    if norm:
        EE = EE/EE.max()
        EEx = EEx/EE.max()
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(data.theta, 10 * np.log10(EE), label=label)
    #ax.plot(data.theta, 10 * np.log10(EEx))
    ax.grid(axis="y")
    #ax.set_ylim([-100, 0])
    ax.set_xlabel("thetas")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    return ax

#######################################
# Métricas para ganho da antena.
#######################################


def I1(data):
    # ref Pontoppidan
    Eco2 = data.Eco ** 2 + data.ImEco ** 2
    Ecx2 = data.Ecx ** 2 + data.ImEcx ** 2
    EE2 = Eco2 + Ecx2
    I1 = 2 * np.pi * np.trapz(EE2 * np.cos(data.theta), data.theta)
    return I1


def I2(data):
    # ref Pontoppidan
    Eco2 = data.Eco ** 2 + data.ImEco ** 2
    I2 = 2 * np.pi * np.trapz(Eco2 * np.cos(data.theta), data.theta)
    return I2


def I3(data):
    # ref Pontoppidan
    Eco = np.sqrt(data.Eco ** 2 + data.ImEco ** 2)
    I3 = 2 * np.pi * np.trapz(Eco * np.cos(data.theta), data.theta)
    return I3


def I4(data):
    # ref Pontoppidan
    ReI4 = np.trapz(data.Eco * np.cos(data.theta), data.theta)
    ImI4 = np.trapz(data.ImEco * np.cos(data.theta), data.theta)
    I4 = 2 * np.pi * np.sqrt(ReI4 ** 2 + ImI4 ** 2)
    return I4


def n_spill(data):
    # ref Pontoppidan
    result = I1(data) / 4 * np.pi
    return result


def n_pol(data):
    # ref Pontoppidan
    result = I2(data) / I1(data)
    return result


def n_amp(data):
    # ref Pontoppidan
    result = I3(data) ** 2 / I2(data)
    return result


def n_phase(data):
    # ref Pontoppidan
    result = I4(data) ** 2 / I3(data)
    return result


def gain_dB(data):
    # ref Pontoppidan
    result = 10 * np.log10(I4(data) ** 2)
    return result


def gain_dB_O(data):
    # ref Olmi
    result = 10 * np.log10(I4(data) ** 2 / I2(data))
    return result


def gain_max(data):
    # ref Gan
    idx = (data.Eco**2 + data.ImEco**2 + data.Ecx**2 + data.ImEcx**2).argmax()
    Bmax = data.iloc[idx].Eco ** 2 + data.iloc[idx].ImEco ** 2 + data.iloc[idx].Ecx ** 2 + data.iloc[idx].ImEcx ** 2
    result = 10 * np.log10(2 * Bmax / I1(data))
    return result

#######################################
# Funções para controle do GRASP.
#######################################


def run_grasp(tor_file):
    """ Roda Grasp a partir de tor file.

    Cria arquivos gxp e tci.
    Muda diretório corrente.
    Executa grasp.
    retorna ao diretório corrente.
    """
    cwd = os.getcwd()
    dst = os.path.abspath(os.path.split(tor_file)[0])
    filename = os.path.abspath(os.path.split(tor_file)[-1]).\
        split(".")[0].split("/")[-1]
    batch_orig = "../grasp/STANDARD/batch.gxp"
    tci_orig = "../grasp/STANDARD/BINGO_SIMPLES.tci"
    batch_file = os.path.abspath(os.path.join(os.path.split(tor_file)[0],
                                              "batch.gxp"))
    tci_file = os.path.abspath(os.path.join(os.path.split(tor_file)[0],
                                            filename + ".tci"))
    # Copia batch file padrão e tci para diretório.
    try:
        shutil.copyfile(batch_orig, batch_file)
        shutil.copyfile(tci_orig, tci_file)
    except:
        print("Erro copiando batch.gxp")
        return
    # Edita batch para nomes do tor.
    with open(batch_file, 'r') as file:
        filedata = file.read()
        filedata = filedata.replace('BINGO_SIMPLES.tor', filename + ".tor")
        filedata = filedata.replace('BINGO_SIMPLES.tci', filename + ".tci")
    with open(batch_file, 'w') as file:
        file.write(filedata)
    # troca de diretório
    os.chdir(dst)
    # Executa grasp
    command = "grasp-analysis batch.gxp run.out run.log"
    process = run_daemon(thread=run_command, command=command)
    # retorna diretório
    os.chdir(cwd)
    return process


def rotate_feed(axis, angle):
    if axis == "x":
        dict = "corneta_coor  coor_sys\n(\n" + \
               " origin           : struct(x: 0.0 cm, y: 0.0 m, z: 0.0 m),\n" +\
               " x_axis           : struct(x: 1.0, y: 0.0, z: 0.0),\n" + \
               " y_axis           : struct(x: 0.0, y: " +\
               str(np.cos(np.radians(angle))) + ", z: " +\
               str(-np.sin(np.radians(angle))) + "),\n" +\
               " base             : ref(dual_feed_coor)\n)"
    elif axis == "y":
        dict = "corneta_coor  coor_sys\n(\n" +\
               " origin           : struct(x: 0.0 cm, y: 0.0 m, z: 0.0 m),\n" +\
               " x_axis           : struct(x: " +\
               str(np.cos(np.radians(angle))) +\
               ", y: 0.0, z: " + str(-np.sin(np.radians(angle))) + "),\n" +\
               " y_axis           : struct(x: 0.0, y: 1.0, z: 0.0),\n" + \
               " base             : ref(dual_feed_coor)\n)"
    return dict


def translate_feed(coord, displacement):
    if coord == "x":
        dict = "corneta_coor  coor_sys\n(\norigin           : struct(x: " + \
                str(displacement) + \
                " cm, y: 0.0 m, z: 0.0 m),\nbase             : ref(dual_feed_coor)\n)\n"
    if coord == "y":
        dict = "corneta_coor  coor_sys\n(\norigin           : struct(x: 0.0 cm, y: " + str(displacement) + " cm, z: 0.0 m),\nbase             : ref(dual_feed_coor)\n)\n"
    if coord == "z":
        dict = "corneta_coor  coor_sys\n(\norigin           : struct(x: 0.0 cm, y: 0.0 cm, z: " + str(displacement) + " cm),\nbase             : ref(dual_feed_coor)\n)\n"
    return dict


def rotate_secundary(axis, angle):
    pass


def translate_secondary(coord, displacement):
    pass

def _make_tor_feed(filename, string):
    # le TOR padrão
    with open("../grasp/STANDARD/BINGO_01.tor", 'r') as file_in:
        lines = file_in.readlines()
    # cria TOR em nova pasta com feed removido.
    with open(filename, "w") as file_out:
        file_out.writelines(lines[:-5])
        file_out.writelines(string)
    return


def make_tor(object="feed", type="rotation", coord="x", value=0):
    dirname = "../grasp/STANDARD/job" + "_" + object + "_" + type + \
        "_" + str(coord) + "_" + str(value) + "/"
    # cria pasta
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    filename = dirname + "bingo.tor"
    if object == "feed":
        if type == "translation":
            # cria string para feed:
            feed_string = translate_feed(coord, value)
        elif type == "rotation":
            feed_string = rotate_feed(coord, value)
    # grava arquivo tor
    _make_tor_feed(filename, feed_string)
    print("Arquivo {} gerado com sucesso.".format(filename))
    return filename


def make_tor_range(object="feed", type="rotation", **kwargs):
    pass


#######################################
# Utilidades para processamento paralelo
#######################################

def run_command(command: str):
    """Executa comando shell passado como argumento utilizando biblioteca `subprocess`.

    Função reporta erro no log.

    Args:
        command (str): string de comando escrita da mesma forma que se escreveria em linha de comando.

    Returns:
        tuple[str, str]: saída padrão e código de erro informado para o comando executado.
    """
    process = subprocess.Popen(
                                shlex.split(command),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                                )
    out, err = process.communicate()
    if err:
        err = err.decode("utf-8")
        logger.error("Error running command {}:{}".format(command, err))
    return (out, err)


def run_daemon(thread=None, *args, **kwargs):
    """Roda o comando indicado em modo background, liberando a execução do resto do programa.

    Retorna o processo em execução.

    """
    try:
        process = multiprocessing.Process(
                                          target=thread,
                                          args=args,
                                          kwargs=kwargs,
                                          daemon=True
                                          )
        process.start()
    except OSError as error:
        logger.error("Detached program {} failed to execute: {}".format(command, error))
    return process
