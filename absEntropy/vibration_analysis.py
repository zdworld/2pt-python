import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
import warnings


# 多原子流体的权重函数部分
# 固态分量
def solid_S0(delta_freq: float, dos_0: float, Temperature: float) -> float:
    '''
    计算固态分量在0频下的振动分量, 频率保存的步长为delta_freq, dos_0为0频态密度, Temperature为特征温度
    返回0频率到第一个频率的积分结果, 由于DOS使用fft只保留了正频率, 且将DOS*2
    0频率和奈奎斯特频率没有对应的负频率, 其贡献应当减半
    '''
    kb = 1.380658e-23
    h = 6.62606896e-34
    c = 2.99792458e10
    u = delta_freq*(h*c)/(kb*Temperature)  # 频率换算为Hz, u为频率单位hv/kT

    def ws_func(u, S0):
        '''
        权重函数, 其中u为hv/kT
        '''
        w = (u/(np.exp(u)-1)-np.log(1-np.exp(-u)))*S0
        return w

    int_result, error = integrate.quad(ws_func, 0, u, args=(dos_0,))

    if error > 1e-6:
        warnings.warn(f"Integral error is too large: {error}")

    return int_result


def solid_S(freq: np.ndarray, Temperature: float):
    '''
    计算固态分量在给定频率下的权重函数,不包含0频分量
    '''
    kb = 1.380658e-23
    h = 6.62606896e-34
    c = 2.99792458e10
    u = freq*(h*c)/(kb*Temperature)  # 频率换算为Hz, u为频率单位hv/kT

    ws = u/(np.exp(u)-1)-np.log(1-np.exp(-u))
    return ws

 # 气体分量


def gas_S_trans(n_mols: int, masses: np.ndarray, temperature: float, volume: float, fluid: float, Delta: float):
    '''
    计算气体分量的平动权重函数
    给出气体部分的平动分量的权重函数, 平动分量使用理想气体熵, 并使用硬球熵校正
    输入参数:n_mols:分子数, masses:总质量, temperature:平动特征温度, volume:体积, fluid:流动性分数, Delta:归一化扩散系数
    '''
    kb = 1.380658e-23
    h = 6.62606896e-34
    Na = 6.0221367e23

    mass = np.sum(masses)/n_mols * 1e-3
    T = temperature
    V = volume*1e-30
    N = n_mols
    y = fluid**2.5/Delta**1.5
    z = (1+y+y**2-y**3)/(1-y)**3    # Carnahan-Starling equation

    gas_ideal = 2.5 + \
        np.log((2*np.pi*mass*kb*T / (Na*h**2))**1.5 * (V/(N*fluid))*z)
    HS_correction = y*(3*y-4)/(1-y)**2

    gas_tran = (gas_ideal + HS_correction)/3
    return gas_tran


def gas_S_rot(symmetry: int, character_Temperature: np.ndarray, Temperature: float, isLiner: bool):
    '''
    气体分量的转动权重函数, 使用刚性转子
    其权重函数依赖于分子形状,线形分子具有不同的配分函数
    输入:symmetry:分子的对称性, character_Temperature:三主轴的转动特征温度, Temperature:转动温度
    '''

    s = symmetry
    T = Temperature
    Tc = character_Temperature

    if isLiner:
        if Tc[0] != 0:
            warnings.warn(
                'Liner molecule must have same Tc2 and Tc3, and The Tc0 must be 0, and this group will be calculated as non-liner molecule')
            gas_rot = np.log((np.e**1.5 * np.pi**0.5 / s) *
                             np.sqrt(T**3/(Tc[0]*Tc[1]*Tc[2])))/3
        else:
            gas_rot = np.log((np.e * T)/(s * np.sqrt(Tc[1]*Tc[2])))/2
    else:
        # 非线性分子
        gas_rot = np.log((np.e**1.5 * np.pi**0.5 / s) *
                         np.sqrt(T**3/(Tc[0]*Tc[1]*Tc[2])))/3

    return gas_rot


def fluid_factor(S0: float, masses: np.ndarray, n_mols: int, temperature: float, density: float):
    '''
    计算流动性分数,输入态密度函数(DOS, cm)和频率(cm^-1), 输出流动性分数
    首先需要计算归一化扩散系数, 再解出流动性因数
    Delta = (2S(0)/9N)*(pi kT/m)^0.5*density^(1/3)*(6/pi)^(2/3)
    density为数密度,这里不使用直接输入的数密度,使用修正值,即每个组的N/V,需要开始K_deltaV与1偏差不太大
    '''
    S0 = S0 / \
        2.99792458  # 换算成标准时间单位,原本是cm,除以光速2.99792458e10 cm/s, 后面乘以Volume^{1/3}，以A为单位是10^-10，刚好抵消
    mass = 1e-3*np.sum(masses) / n_mols  # 求出每个分子的摩尔质量,这里要求每个分组必须是同类分子
    Delta = (2*S0 / (9*n_mols))*(np.pi*8.3138462*temperature /
                                 mass)**(1/2)*(6/np.pi)**(2/3)*density**(1/3)

    def fluid_eq(f):
        D = Delta
        return 2*D**(-9/2)*f**(15/2)-6*D**(-3)*f**5-D**(-3/2)*f**(7/2)+6*D**(-3/2)*f**(5/2)+2*f-2

    initial_f = np.array([min(0.5, 0.7293*Delta**0.5727)])
    solution = fsolve(func=fluid_eq, x0=initial_f, xtol=1e-7, maxfev=1000)
    return solution[0], Delta


def dos_gaslike(S0: float, freq: np.ndarray, f: float, n_mols: int, DOS: np.ndarray):
    '''
    DOS单位使用cm, 返回一个gas-like的dos,从总dos中减去gas分量得到固态分量
    '''
    DOS_gas = S0 / (1 + ((np.pi * S0 * freq) / (6 * f * n_mols))**2)
    DOS_gas = np.minimum(DOS_gas, DOS)
    return DOS_gas
