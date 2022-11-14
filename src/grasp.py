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
import scipy as sp

#######################################
# Funções para manipulação de cuts
#######################################
GRASP = "grasp-analysis"

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
    eta_spill = []
    eta_pol = []
    eta_amp = []
    eta_phase = []
    gains_max = []
    FWHMs = []
    params = []
    for file in cut_files:
        df = read_cut(file)
        dfs.append(df)
        param = list(filter(lambda val: val != 0,
                            [float(val) for val in
                             file.split("/")[-2].split("_")[-5:]]))
        if len(param) == 0:
            param = 0.
        else:
            param = param[0]
        df = center_beam(df)
        params.append(param)
        FWHM = get_FWHM(df)
        FWHMs.append(FWHM)
        gains.append(gain(df, FWHM))
        eta_spill.append(n_spill(df, FWHM))
        eta_pol.append(n_pol(df, FWHM))
        eta_phase.append(n_phase(df, FWHM))
        eta_amp.append(n_amp(df, FWHM))
        gains_max.append(gain_max(df, FWHM))
    df_gains = pd.DataFrame({"Values": params, "FWHM": FWHMs,
                             "Gain": gains,
                             "eta_spill": eta_spill,
                             "eta_pol": eta_pol,
                             "eta_amp": eta_amp,
                             "eta_phase": eta_phase,
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
    ax.set_xlabel(r"$\theta (^\circ)$")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    return ax

def normalize(data):
    columns = ["Gain", "eta_spill", "eta_pol", "eta_amp", "eta_phase", "Gain_Max"]
    zero_values = data[data.Values == 0]
    for column in columns:
        new_columns = "delta_" + column
        data[new_columns] = np.abs(data[column] - zero_values[column].values)/ zero_values[column].values
    return data.sort_values(by="Values")

#######################################
# Utilidades para padrão de radiação
#######################################


def get_B(data, type="co"):
    Bco = data.Eco ** 2 + data.ImEco ** 2
    Bcx = data.Ecx ** 2 + data.ImEcx ** 2
    BB = Bco + Bcx
    if type == "co":
        result = Bco
    elif type == "cx":
        result = Bcx
    elif type == "total":
        result = BB
    return result


def _interpolate_beam(x, data):
    BB = get_B(data, type="total")
    result = sp.interpolate.interp1d(data.theta, BB)
    return -result(x)


def get_FWHM(data, theta0=.1):
    BB = get_B(data, type="total")
    FWHM_func = lambda theta: _interpolate_beam(theta, data) +\
        BB.max() / 2
    FWHM = sp.optimize.root(FWHM_func, theta0, method="lm").x[0]
    return FWHM

#######################################
# Métricas para ganho da antena.
#######################################
def center_beam(data):
    df = data.copy()
    BB = get_B(df, type="total")
    idx = BB.argmax()
    theta_c = df.iloc[idx].theta
    df.theta = df.theta - theta_c
    return df

def I1(data, theta_max):
    # ref Pontoppidan
    df = data.copy()
    df = df[(df.theta >= 0) & (df.theta <= theta_max)]
    BB = get_B(df, type="total")
    I1 = 2 * np.pi * np.trapz(BB * np.sin(np.radians(df.theta)),
                              np.radians(df.theta))
    return I1


def I2(data, theta_max):
    # ref Pontoppidan
    df = data.copy()
    df = df[(df.theta >= 0) & (df.theta <= theta_max)]
    BBco = get_B(df, type="co")
    I2 = 2 * np.pi * np.trapz(BBco * np.sin(np.radians(df.theta)),
                              np.radians(df.theta))
    return I2


def I3(data, theta_max):
    # ref Pontoppidan
    df = data.copy()
    df = df[(df.theta >= 0) & (df.theta <= theta_max)]
    EEco = np.sqrt(get_B(df, type="co"))
    I3 = 2 * np.pi * np.trapz(EEco * np.sin(np.radians(df.theta)),
                              np.radians(df.theta))
    return I3


def I4(data, theta_max):
    # ref Pontoppidan
    df = data.copy()
    df = df[(df.theta >= 0) & (df.theta <= theta_max)]
    ReI4 = 2 * np.pi * np.trapz(df.Eco * np.sin(np.radians(df.theta)),
                                np.radians(df.theta))
    ImI4 = 2 * np.pi * np.trapz(df.ImEco * np.sin(np.radians(df.theta)),
                                np.radians(df.theta))
    I4 = np.sqrt(ReI4 ** 2 + ImI4 ** 2)
    return I4


def n_spill(data, FWHM):
    # ref Pontoppidan
    result = I1(data, FWHM) / I1(data, np.pi)
    return result


def n_pol(data, FWHM):
    # ref Pontoppidan
    result = I2(data, FWHM) / I1(data, FWHM)
    return result


def n_amp(data, FWHM):
    # ref Pontoppidan
    Omega = 2 * np.pi * (1 - np.cos(np.radians(FWHM)))
    result = I3(data, FWHM) ** 2 / (I2(data, FWHM) * Omega)
    return result


def n_phase(data, FWHM):
    # ref Pontoppidan
    result = I4(data, FWHM) ** 2 / I3(data, FWHM) ** 2
    return result


def gain(data, FWHM):
    # ref Pontoppidan
    Omega = 2 * np.pi * (1 - np.cos(np.radians(FWHM)))
    result = I4(data, FWHM) ** 2 / (I1(data, np.pi) * Omega)
    return result


def gain_max(data, FWHM):
    # ref Gan
    BB = get_B(data, type="total")
    result = 10 * np.log10(4 * np.pi * BB.max() / I1(data, np.pi))
    return result

#######################################
# Funções para controle do GRASP.
#######################################


def run_grasp(tor_file, gpxfile="../grasp/STANDARD/batch.gxp", tcifile="../grasp/STANDARD/BINGO_SIMPLES.tci", daemon=True):
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
    batch_orig = gpxfile
    tci_orig = tcifile
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
    command = GRASP + " batch.gxp " + filename + ".out " + filename +\
        ".log"
    if daemon:
        result = run_daemon(thread=run_command, command=command)
    else:
        result = run_command(command)
    # retorna diretório
    os.chdir(cwd)
    return result


def move_feed(x=0., y=0., z=0., theta=0., phi=0.):
    x0 = x
    y0 = y
    z0 = z
    xx = np.round(np.cos(np.radians(theta)), 4)
    xy = np.round(np.sin(np.radians(theta)) * np.sin(np.radians(phi)), 4)
    xz = np.round(np.sin(np.radians(theta)) * np.cos(np.radians(phi)), 4)
    yx = 0.
    yy = np.round(np.cos(np.radians(phi)), 4)
    yz = np.round(-np.sin(np.radians(phi)), 4)
    dict = "corneta_coor  coor_sys\n(\n  origin           : struct(x: " + \
        str(x0) + " m, y: " + str(y0) + " m, z: " + str(z0) + " m),\n" + \
        "  x_axis           : struct(x: " +\
        str(xx) + ", y: " + str(xy) + ", z: " + str(xz) + "),\n" +\
        "  y_axis           : struct(x: " +\
        str(yx) + ", y: " + str(yy) + ", z: " + str(yz) + "),\n" +\
        "  base             : ref(dual_feed_coor)\n)\n"
    return dict


def rotate_secondary(axis, angle):
    pass


def translate_secondary(x=0, y=0, z=0):
    x0 = 226.5414993 + x
    y0 = 0 + y
    z0 = 48.3552713 + z
    dict = "dual_sub_coor  coor_sys\n(\n  origin           : struct(x: " +\
        str(x0) + " m, y: " + str(y0) + " m, z: " + str(z0) + " m),\n" + \
        "  x_axis           : struct(x: 0.871557553071891E-01, y: 0.0, z: -0.996194696992929),\n" +\
        "  y_axis           : struct(x: 0.0, y: -1.0, z: 0.0),\n" +\
        "  base             : ref(dual_cut_coor)\n)\n"
    return dict


def _make_tor(filename, string, torfile, idx_i, idx_f):
    # le TOR padrão
    with open(torfile, 'r') as file_in:
        lines = file_in.readlines()
    # cria TOR em nova pasta com feed removido.
    with open(filename, "w") as file_out:
        file_out.writelines(lines[:-idx_i])
        file_out.writelines(string)
        if idx_f > 1:
            file_out.writelines("\n")
            file_out.writelines(lines[-idx_f:])
    with open(filename, 'r') as file:
        filedata = file.read()
        datafile = filename.split(".")[-2].split("/")[-1] + ".cut"
        filedata = filedata.replace('dual_cut.cut', datafile)
    with open(filename, 'w') as file:
        file.write(filedata)
    return


def make_tor(object="feed", X0=0, Y0=0, Z0=0, theta=0, phi=0,
             torfile="../grasp/STANDARD/BINGO_CUT.tor", verbose=True,
             prefix="grasp"):
    dirname = "../data/raw/grasp/job" + prefix + "_" + object + "_" + str(X0) +\
        "_" + str(Y0) + "_" + str(Z0) + "_" + str(theta) + "_" + str(phi) + "/"
    # cria pasta
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    filename = dirname + "bingo.tor"
    if object == "feed":
        feed_string = move_feed(x=X0, y=Y0, z=Z0, theta=theta, phi=phi)
        # grava arquivo tor
        _make_tor(filename, feed_string, torfile, 15, 7)
    elif object == "secondary":
        sec_string = translate_secondary(x=X0, y=Y0, z=Z0)
        _make_tor(filename, sec_string, torfile, 7, 0)
    if verbose:
        print("Arquivo {} gerado com sucesso.".format(filename))
    return filename

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
