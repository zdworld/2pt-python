
import math
import numpy as np
from scipy.signal import correlate
from scipy import fft
import pyfftw
fft.set_global_backend(pyfftw.interfaces.scipy_fft)


# 三斜晶胞体积
def volume_triclinic(a, b, c, alpha, beta, gamma):
    # 将角度从度转换为弧度
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    # 计算体积
    volume = (a * b * c) * math.sqrt(
        1 + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad) -
        math.cos(alpha_rad)**2 -
        math.cos(beta_rad)**2 -
        math.cos(gamma_rad)**2
    )

    return volume


# 计算速度的质量加权自相关函数
def vacf_mass(vel: np.ndarray, masses: np.ndarray):
    '''
    计算速度的自相关函数, 每个维度分量计算完成后求和, 再对质量加权求和
    输入为(n_frames, n_atoms, 3)的速度矩阵
    返回已经三方向求和的质量加权自相关函数(n_frames, n_atoms)
    '''
    n_frames, n_atoms, _ = vel.shape
    # 速度矩阵展平用于计算自相关函数矩阵
    # vel = vel.reshape(n_frames, n_atoms * 3) - np.mean(vel.reshape(n_frames, n_atoms * 3), axis=0)
    vel = vel.reshape(n_frames, n_atoms * 3)
    vacf = np.zeros((2*n_frames-1, n_atoms * 3), dtype="float32")

    # 对第1个维度进行自相关计算
    vacf = np.apply_along_axis(lambda x: correlate(
        x, x, mode="full", method="fft"), axis=0, arr=vel)
    # 仅保留后半部分
    vacf = vacf[n_frames-1:].reshape(-1, n_atoms, 3) / vel.shape[0]
    # vacf = vacf.reshape(-1, n_atoms, 3) / vacf.shape[0]
    vacf_sum3d = np.sum(vacf, axis=2) * masses[np.newaxis, :]

    return vacf_sum3d

# # 计算速度的质量加权自相关函数
# def vacf_mass(vel: np.ndarray, masses: np.ndarray):
#     '''
#     计算速度的自相关函数, 每个维度分量计算完成后求和, 再对质量加权求和
#     输入为(n_frames, n_atoms, 3)的速度矩阵
#     返回已经三方向求和的质量加权自相关函数(n_frames, n_atoms)
#     '''
#     n_frames, n_atoms, _ = vel.shape
#     # 速度矩阵展平用于计算自相关函数矩阵
#     vel = vel.reshape(n_frames, n_atoms * 3)

#     vacf_sum3d = np.zeros((n_frames//2, n_atoms), dtype="float32")
#     vel_fft = np.zeros(vel.shape, dtype="complex64")
#     vel_ifft = np.zeros(vel.shape, dtype="complex64")

#     vel_fft = fft.fft(vel, axis=0, norm="backward", workers=16)
#     vel_ifft = np.real(vel_fft * vel_fft.conj())
#     vel_ifft = fft.ifft(vel_ifft, axis=0, norm="forward", workers=16, overwrite_x=True) / n_frames
#     vacf_sum3d = np.sum(np.real(vel_ifft.reshape(-1, n_atoms, 3)), axis=2)[:n_frames//2, :] * masses[np.newaxis, :] / n_frames

#     return vacf_sum3d



def vacf_omega(omega: np.ndarray):
    '''
    角速度的态密度计算
    输入:omega,(n_frames,n_mols,3)转动惯量均方根加权的角速度
        inertia_principle,(n_frames,n_mols,3)三个惯性主轴的转动惯量
    返回:vacf_omega,(n_frames, nmols)为每个分子三方向角速度对sqrt(inertia_principle)加权后的自相关函数
    '''
    n_frames, n_mols, _ = omega.shape
    vacf = np.zeros((2*n_frames-1, n_mols*3), dtype="float32")
    omega_wby_I = omega.reshape(n_frames, n_mols*3)
    # 对第一维进行自相关函数计算
    vacf = np.apply_along_axis(lambda x: correlate(
        x, x, mode="full", method="fft"), axis=0, arr=omega_wby_I)
    vacf = vacf[n_frames-1:].reshape(-1, n_mols, 3) / omega.shape[0]
    # 角速度已经对惯量的平方根加权，直接求和
    vacf_sum3d = np.sum(vacf, axis=2)

    return vacf_sum3d

# def vacf_omega(omega: np.ndarray):
#     '''
#     角速度的态密度计算
#     输入:omega,(n_frames,n_mols,3)转动惯量均方根加权的角速度
#         inertia_principle,(n_frames,n_mols,3)三个惯性主轴的转动惯量
#     返回:vacf_omega,(n_frames, nmols)为每个分子三方向角速度对sqrt(inertia_principle)加权后的自相关函数
#     '''
#     n_frames, n_mols, _ = omega.shape
#     # 速度矩阵展平用于计算自相关函数矩阵
#     omega = omega.reshape(n_frames, n_mols * 3)
#     vacf_sum3d = np.zeros((n_frames//2, n_mols), dtype="float32")
#     omega_fft = np.zeros(omega.shape, dtype="complex64")
#     omega_ifft = np.zeros(omega.shape, dtype="complex64")

#     omega_fft = fft.fft(omega, axis=0, norm="backward", workers=16)
#     omega_ifft = np.real(omega_fft * omega_fft.conj())
#     omega_ifft = fft.ifft(omega_ifft, axis=0, norm="forward", workers=16, overwrite_x=True) / n_frames
#     vacf_sum3d = np.sum(np.real(omega_ifft.reshape(-1, n_mols, 3)), axis=2)[:n_frames//2, :] / n_frames
#     return vacf_sum3d


# 速度自相关函数的傅里叶变换
def vacf_fft(vacf: np.ndarray, dt):
    fft_vacf = np.zeros(vacf.shape, dtype="float32")
    fft_vacf = np.real(fft.fft(vacf, axis=0, norm="backward", workers=4))
    fft_vacf = fft_vacf[:vacf.shape[0]//2, :]
    # 以cm^-1为单位,1 THz = 33.3564 cm^{-1},采样间隔1ps即为最大采样频率1THz
    freq_vacf = 33.3564*fft.fftfreq(vacf.shape[0], dt)[:vacf.shape[0]//2]
    return fft_vacf, freq_vacf
