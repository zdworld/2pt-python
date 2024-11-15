import MDAnalysis as mda
import numpy as np


import gc
from typing import List
from functools import reduce
from itertools import combinations
import yaml
from .velocity_analysis import calcu_v_trans, calcu_v_rot, kieticE_aver
from .vacf import *
from .vibration_analysis import *
import pandas as pd
import warnings


# 定义分子分组和自由度信息
class AtomGroups:
    def __init__(self, inputfile):
        '''
        读取配置文件里分组部分的信息, 检查选择语句和分组的合法性
        '''
        with open(inputfile, 'r') as f:
            settings: dict = yaml.load(f, Loader=yaml.FullLoader)

        u = mda.Universe(settings['Input']['topology'],
                         settings['Input']['trajectory'], in_memory=False)
        self.n_groups = len(settings['Groups'])
        self._volume = volume_triclinic(*u.trajectory[-1].dimensions)
        self.dt = u.trajectory.dt
        # 检测groups合法性, 选择时会包含选择语句选择的那部分
        group_indices: List[np.ndarray] = []
        for group in settings['Groups'].values():
            try:
                atom_sel = u.select_atoms(group['selection'])
                atom_grp = atom_sel.residues.atoms          # selection总会选中整个残基
                group_indices.append(atom_grp.indices)
            except:
                warnings.warn(
                    f"Selection {group['selection']} is invalid, please check the selection statement.")
                exit()
        # 检查是否有重复的原子
        if len(group_indices) > 1:
            common_element: List = []
            for grp1, grp2 in combinations(group_indices, 2):
                common = np.intersect1d(grp1, grp2)
                if common.size != 0:
                    common_element.append(common)
            if len(common_element) != 0:
                repeated_atoms = sum((list(arr) for arr in common_element), [])
                warnings.warn(
                    f"Total {len(set(repeated_atoms))} atoms are not unique. This can lead to a wrong result. Please check the selection statements.")
                exit()
                
        # 检查总和是否为所有原子
        union_element = reduce(np.union1d, group_indices)
        is_equal = np.array_equal(
            np.sort(union_element), np.sort(u.atoms.indices))
        if not is_equal:
            warnings.warn(
                f"Selections do not cover all atoms. This can lead to a wrong result.")
            exit()
        # 检查无误, 计算部分基本属性
        self.Groups = settings['Groups']
        for group in self.Groups.values():
            group['index'] = u.select_atoms(group['selection']).indices
            group['mass'] = u.select_atoms(group['selection']).masses
            group['n_atoms'] = len(u.select_atoms(group['selection']).atoms)
            group['n_mols'] = len(u.select_atoms(group['selection']).fragments)
            group['constraints'] = group['constraints'] * group['n_mols']
            group['velocities'] = np.zeros(
                # 平动。转动，振动
                (3, len(u.trajectory), group['n_mols'], group['n_atoms']//group['n_mols'], 3), dtype="float32")
            group['omega'] = np.zeros(
                # 角速度
                (len(u.trajectory), group['n_mols'], 3), dtype="float32")
            group['omega I'] = np.zeros(
                # 惯量加权
                (len(u.trajectory), group['n_mols'], 3), dtype="float32")
            group['rot temperature'] = np.zeros((3))  # 转动特征温度

            # 假设怕平动振动转动温度相同,5项分别是trans,rot,vib,omega,average, 且omega具有与rot相同的值
            group['Temperature'] = [settings['Input']['Temperature']]*5
            # 预分配存放vacf矩阵的位置,(n_frames,5),5为tra,rot,vib,omega.total
            group['vacf'] = np.zeros((len(u.trajectory), 5), dtype="float32")
            # group['vacf'] = np.zeros((len(u.trajectory)//2, 5), dtype="float32")
            # 多一列，为角速度的态密度,5列依次为tra,rot,vib,omega,total
            group['DOS'] = np.zeros((group['vacf'].shape[0]//2, 5), dtype="float32")
            group['DOS 2PT'] = {"translation": np.zeros((group['DOS'].shape[0], 2), dtype="float32"),
                                "rotation": np.zeros((group['DOS'].shape[0], 2), dtype="float32"),
                                "vibration": np.zeros((group['DOS'].shape[0], 1), dtype="float32"),
                                # 初始化2pt分解后的dos，平动转动各2列，气相和固相。振动没有气体分量，只有1列
                                "omega": np.zeros((group['DOS'].shape[0], 2), dtype="float32")}
            group['fluidicity'] = [0., 0., 0.]  # 流动性因子,tran,rot,omega
            group['Delta'] = [0., 0., 0.]       # 无量纲流动性因子,tran,rot,omega
            group['diffusion'] = [0., 0., 0., 0.]  # 平动，转动，振动，角速度对应的S0求得的扩散系数
            group['Entropy'] = {"translation": [0., 0.],
                                "rotation": [0., 0.],
                                "vibration": [0., 0.],
                                }               # 熵,平动转动振动分为gas-like分量和solid-like分量, vib无gas分量

            # 根据原子数与分子数是否相等判定是否为单原子流体,单原子流体无转动振动
            # 自由度计算,约束从振动自由度中扣除,线形分子的转动自由度为2
            if group['n_atoms'] == group['n_mols']:
                group['isMonatomic'] = True
                group['DOF'] = (3*group['n_atoms'], 0, 0)  # 分别为平动,转动,振动
                if group['constraints'] != 0:
                    warnings.warn(
                        'Monatomic fluid with a nonzero constraint. This indicates an error in the constraints or a topological error. Please check the configuration file and topology.')
                    exit()
                    return
            else:
                group['isMonatomic'] = False
                # 计算振动自由度,遍历每个片段,振动自由度为3N-6,再在总值中扣除约束, 线性分子(如二氧化碳,以及双原子分子)只有两个转动自由度,振动自由度为3N-5
                if group['isLiner'] == False:
                    DOF_vib = 0
                    for frag in u.select_atoms(group['selection']).fragments:
                        DOF_vib += 3*len(frag.atoms) - 6
                    group['DOF'] = (
                        3*group['n_mols'], 3*group['n_mols'], DOF_vib - group['constraints'])
                elif group['isLiner'] == True:
                    DOF_vib = 0
                    for frag in u.select_atoms(group['selection']).fragments:
                        DOF_vib += 3*len(frag.atoms) - 5
                    group['DOF'] = (
                        3*group['n_mols'], 2*group['n_mols'], DOF_vib - group['constraints'])
        # 计算体积分量,单纯按摩尔分数分配是显而易见的不合理的,应当按实际偏摩尔量计算,但Kirkwood–Buff方案收敛困难
        # 此处使用数密度估算,存在一定的误差,但精度已经相对较为可靠,参考：Phys. Chem. Chem. Phys., 2012, 14, 15206–15213
        # 缩放系数,反映混合物对理想混合物的偏差,偏差越大估计误差越大
        k_deltaV = np.sum(np.array([grp['n_mols']/grp['density']
                          for grp in self.Groups.values()]))*1000/self._volume
        print(f'K_deltaV = {k_deltaV:.4f} \nIf there is a large deviation to 1, please check the density(nm^-3) of each group.')
        for group in self.Groups.values():
            group['volume'] = 1000*group['n_mols']/(group['density']*k_deltaV)

    def remove_groups(self):
        '''
        移除不被计算的group, 这些group的作用是计算每个group的体积分量,但并不参与计算
        '''
        to_rm = []
        for key in self.Groups.keys():
            group = self.Groups[key]
            if group['isCalculated'] == False:
                print(f"Group {group['name']} is not calculated, removing it...")
                to_rm.append(key)
        for key in to_rm:
            self.Groups.pop(key)
        gc.collect()


class DOSdist:
    def __init__(self, inputfile, groupinfo: AtomGroups):
        with open(inputfile, 'r') as f:
            settings: dict = yaml.load(f, Loader=yaml.FullLoader)
        print("Reading trajectory...")
        self.universe = mda.Universe(
            settings['Input']['topology'], settings['Input']['trajectory'], in_memory=True)
        print('Trajectory has been read into memory successfully!')
        self.grps_info = groupinfo
        # 定义输出
        self.outfiles = settings['Output']

    def print_information(self):
        print("TRAJ INFOMATION:")
        print(f"Atomes      : {len(self.universe.atoms)}")
        print(f"Molecules   : {self.universe.atoms.n_fragments}")
        print(f"Traj Freames: {len(self.universe.trajectory)}")
        print(f"Begin Time  : {self.universe.trajectory[0].time:.3f} ps")
        print(f"End Time    : {self.universe.trajectory[-1].time:.3f} ps")
        print(f"Traj Step   : {self.universe.trajectory.dt:.3f} ps")
        # 查看盒子尺寸（周期性边界条件）
        print(
            f"Box Size    : x, y, z = {self.universe.trajectory[-1].dimensions[0]:.2f}, {self.universe.trajectory[-1].dimensions[1]:.2f}, {self.universe.trajectory[-1].dimensions[2]:.2f} A")
        print(
            f"              alpha, beta, gamma = {self.universe.trajectory[-1].dimensions[3]:.2f}, {self.universe.trajectory[-1].dimensions[4]:.2f}, {self.universe.trajectory[-1].dimensions[5]:.2f} degeree")
        print(
            f"Volume      : {volume_triclinic(*self.universe.trajectory[-1].dimensions)} A^3")

    def velocity_analysis(self):
        '''
        速度分解并进行特征温度计算
        '''
        # 计算不同运动的温度,根据能量均分定理,动能会均匀的分布在各个自由度上,因此温度差异不大。
        # 对于低温和高频的情况,能量均分定理近似不再有效,对于这种高频振动,md中通常会直接约束。即此温度下的能量不足以进入高能级,对应的自由度就消失了。
        # 当某个运动分量的特征温度过低,此时应当检查约束的设置是否正确,例如刚性水的振动自由度,双原子分子的振动自由度等。
        # 对每组分别计算并写入group_info
        self._vel = np.array(
            [ts.velocities for ts in self.universe.trajectory])
        self._coord = np.array(
            [ts.positions for ts in self.universe.trajectory])
        for group in self.grps_info.Groups.values():
            print(f"Calculating the velocities of Group:{group['name']}.")
            vel = self._vel[:, group['index'], :]       # group的总速度
            coord = self._coord[:, group['index'], :]   # group的总坐标

            mol_atoms: int = group['n_atoms']//group['n_mols']
            vel = vel.reshape(-1, group['n_mols'], mol_atoms, 3)
            coord = coord.reshape(-1, group['n_mols'], mol_atoms, 3)
            masses = group['mass'][0:mol_atoms]    # 单分子质量

            if group['isMonatomic']:
                # 若为单原子分子，跳过分配过程，无转动和振动速度
                group['velocities'][0, :, :, :] = vel
            else:
                # 多原子非线形分子和线形分子
                # 平动，转动，振动
                group['velocities'][0, :, :, :, :] = calcu_v_trans(vel, masses)
                print("     Translation done...")
                group['velocities'][1, :, :, :, :], group['omega'], group['omega I'], group['rot temperature'] = calcu_v_rot(
                    coord, vel, masses, group['isLiner'])
                print("     Rotation done...")
                group['velocities'][2, :, :, :, :] = vel - group['velocities'][0,
                                                                               :, :, :, :] - group['velocities'][1, :, :, :, :]
                print("     Vibration done...")
            # 形状重整为(3, n_frames, n_atoms, 3)
            group['velocities'] = group['velocities'].reshape(
                3, -1, group['n_atoms'], 3)
            # 删除原速度和坐标解除内存占用
            # 计算每个分量的对应温度
            KE_tra = np.sum(kieticE_aver(
                group['velocities'][0, :, :, :], group['mass']))
            KE_rot = np.sum(kieticE_aver(
                group['velocities'][1, :, :, :], group['mass']))
            KE_vib = np.sum(kieticE_aver(
                group['velocities'][2, :, :, :], group['mass']))
            # 计算每种运动的温度，并计算其自由度
            KE = [KE_tra, KE_rot, KE_vib]
            T_aver: float = 0.0
            for i in range(3):
                if group['DOF'][i] == 0:
                    group['Temperature'][i] = 0
                else:
                    group['Temperature'][i] = 2*KE[i] / \
                        (8.3138462*group['DOF'][i])
            T_aver = (group['Temperature'][0]*group['DOF'][0] + group['Temperature'][1]*group['DOF'][1] + group['Temperature'][2] *
                      group['DOF'][2])/(group['DOF'][0] + group['DOF'][1] + group['DOF'][2])
            group['Temperature'][4] = T_aver
            group['Temperature'][3] = group['Temperature'][1]

            print("     kinetic energy done...")

            print(f"Group:{group['name']} done.")
        print("Velocities have been decomposed.")

    def vacf(self):
        '''
        计算每个group的三组vacf,写入属性
        由于角速度的自相关函数和转动速度的自相关函数完全一致，只计算一组即可
        但计算角速度的态密度时使用了各个主轴的转动惯量, 这会导致态密度不同, 对
        '''
        gc.collect()
        print("Calculating the VACF of each group...")
        # 0-2列为trans,rot,vib; 第3列为omega,最后一列为总的自相关函数
        for group in self.grps_info.Groups.values():
            if group['isMonatomic']:
                # 若为单原子分子, 则只有平动分量
                group['vacf'][:, 0] = np.sum(
                    vacf_mass(group['velocities'][0, :, :, :], group['mass']), axis=1)
                group['vacf'][:, 4] = group['vacf'][:, 0]
            else:
                # 多原子分子合并处理
                for i in range(3):
                    group['vacf'][:, i] = np.sum(
                        vacf_mass(group['velocities'][i, :, :, :], group['mass']), axis=1)

                # 计算角速度的惯量加权
                group['vacf'][:, 3] = np.sum(
                    vacf_omega(group['omega I']), axis=1)
                # 计算总速度
                group['vacf'][:, 4] = np.sum(
                    vacf_mass(np.sum(group['velocities'], axis=0), group['mass']), axis=1)

            print(f"     group {group['name']} done.")
        print("VACF calculation for each group complated. ")

    def dos(self):
        '''
        计算每组每个分量的DOS, 对fft后的结果进行处理, 计算积分值与自由度的差异
        '''
        print('Calculating the work function...')
        # 解除内存占用
        for group in self.grps_info.Groups.values():
            # 解除对速度的占用
            group['velocities'] = None
            gc.collect()
            Temperatures = np.array(group['Temperature'])
            # 加一小值防止除以0
            for i in range(4):
                if Temperatures[i] <= 0:
                    Temperatures[i] = 0.1

            group['vacf'][0, :] = group['vacf'][0, :] * \
                0.5  # 由于只保留了正时延信号,0时延处产生边界问题,截断一半来保持对称
            group['DOS'], self.fft_freq = vacf_fft(
                20*group['vacf']/(8.3138462*Temperatures), dt=self.grps_info.dt)
            # 单位换算至cm,原理是无量纲数乘以dt*v_light，得到单位cm，ps问1e-12 s，cm/s的光速值2.997e10
            group['DOS'] = group['DOS']*self.grps_info.dt*2.99792458e-2
            # 只保留了一半的自相关函数,能量损失一半,乘以2得到总值
            group['DOS'] = group['DOS']*2
            group['vacf'][0, :] = group['vacf'][0, :] * \
                2                                       # 自相关函数复原
            # 将温度为0的分量置为0
            for i in range(4):
                if group['Temperature'][i] == 0:
                    group['DOS'][:, i] = 0.0
            # 若为单原子分子, 则只有平动分量
            if group['isMonatomic']:
                group['DOS'][:, 1:5] = 0.0
            # 根据0频分量计算扩散系数
            DOS_0 = group['DOS'][0, 0:4]
            Temperatures = np.array(group['Temperature'])[0:4]
            for i in range(4):
                group['diffusion'][i] = (
                    DOS_0[i] * 8.3138462 * Temperatures[i] * 1e5)/(12 * 2.99792458e8 * np.sum(group['mass']))

        print('Done!')

    def dos_2pt(self):
        '''
        将平动和转动的DOS分配至gaslike和solidlike两种分量上,同时会得到计算扩散系数和流动性因子
        若K_deltaV偏差过大,结果可能会不可靠,应当更精细的计算摩尔体积并写入输入文件，以得到可靠的数密度
        '''
        print('Calculating the gas-like and solid-likee components ...')
        for group in self.grps_info.Groups.values():
            if group['Temperature'][0] > 0:
                # 处理平动分量的gas-like分量的比重
                S0_tra = group['DOS'][0, 0]
                density = group['n_mols'] / group['volume']
                T_tra = group['Temperature'][0]
                f_tra, Delta_tra = fluid_factor(
                    S0=S0_tra, masses=group['mass'], n_mols=group['n_mols'], temperature=T_tra, density=density)
                # 得到流动性参数f，计算平动的gas_like的DOS,group['DOS 2PT']['translation']第0列为gas，第1列为solid
                if f_tra == 0:
                    group['DOS 2PT']['translation'][:, 1] = group['DOS'][:, 0]
                else:
                    group['DOS 2PT']['translation'][:, 0] = dos_gaslike(
                        S0=group['DOS'][0, 0], freq=self.fft_freq, f=f_tra, n_mols=group['n_mols'], DOS=group['DOS'][:, 0])
                    # group['DOS 2PT']['translation'][:, 1] = group['DOS'][:,
                    #                                                      0] - group['DOS 2PT']['translation'][:, 0]
                    group['DOS 2PT']['translation'][:, 1] = np.clip((group['DOS'][:,
                                                                         0] - group['DOS 2PT']['translation'][:, 0]), 0, None)                
                    group['DOS 2PT']['translation'][0, 1] = 0.0

            if group['Temperature'][1] > 0:
                # 处理转动分量的gas-like分量的比重, 此数值未使用
                S0_rot = group['DOS'][0, 1]
                density = group['n_mols'] / group['volume']
                T_rot = group['Temperature'][1]
                f_rot, Delta_rot = fluid_factor(
                    S0=S0_rot, masses=group['mass'], n_mols=group['n_mols'], temperature=T_rot, density=density)
                # 得到流动性参数f，计算平动的gas_like的DOS
                if f_rot == 0:
                    group['DOS 2PT']['rotation'][:, 1] = group['DOS'][:, 0]
                else:
                    group['DOS 2PT']['rotation'][:, 0] = dos_gaslike(
                        S0=group['DOS'][0, 1], freq=self.fft_freq, f=f_rot, n_mols=group['n_mols'], DOS=group['DOS'][:, 1])
                    group['DOS 2PT']['rotation'][:, 1] = group['DOS'][:,
                                                                      1] - group['DOS 2PT']['rotation'][:, 0]

                    group['DOS 2PT']['rotation'][0, 1] = 0.0

            if group['Temperature'][2] > 0:
                # 处理振动分量，如果特征温度小于等于0，则证明无此分量，无需计算
                group['DOS 2PT']['vibration'] = group['DOS'][:,
                                                             # 振动无气体分量
                                                             2].reshape(-1, 1)

            if group['Temperature'][3] > 0:
                # 处理角速度分量，如果特征温度小于等于0，则证明无此分量，无需计算
                S0_omega = group['DOS'][0, 3]
                density = group['n_mols'] / group['volume']
                T_omega = group['Temperature'][3]
                f_omega, Delta_omega = fluid_factor(
                    S0=S0_omega, masses=group['mass'], n_mols=group['n_mols'], temperature=T_omega, density=density)
                if f_omega == 0:
                    group['DOS 2PT']['omega'][:, 1] = group['DOS'][:, 3]
                else:
                    group['DOS 2PT']['omega'][:, 0] = dos_gaslike(
                        S0=group['DOS'][0, 3], freq=self.fft_freq, f=f_omega, n_mols=group['n_mols'], DOS=group['DOS'][:, 3])
                    # group['DOS 2PT']['omega'][:, 1] = group['DOS'][:,
                    #                                                3] - group['DOS 2PT']['omega'][:, 0]
                    group['DOS 2PT']['omega'][:, 1] = np.clip((group['DOS'][:,
                                                                   3] - group['DOS 2PT']['omega'][:, 0]), 0, None)
                    group['DOS 2PT']['omega'][0, 1] = 0.0

            group['fluidicity'] = [f_tra, f_rot, f_omega]
            group['Delta'] = [Delta_tra, Delta_rot, Delta_omega]

            print("Done.")

    def vibration_analysis(self):
        # dos 2pt中储存了全部的功函数，乘以加权函数即可
        for group in self.grps_info.Groups.values():
            print(f"Analyzing DOS weighting function for group {group['name']}...")
            if group['isMonatomic']:
                # 单原子流体，仅平动分量
                group['Entropy']['translation'] = [0., 0.]
                # 通用参数
                n_mols = group['n_mols']
                masses = group['mass']
                T_tra = group['Temperature'][0]
                f_tra = group['fluidicity'][0]
                Delta_tra = group['Delta'][0]
                volume = group['volume']
                delta_freq = self.fft_freq[1] - self.fft_freq[0]

                gas_weight = gas_S_trans(
                    n_mols, masses, T_tra, volume, f_tra, Delta_tra)
                solid_weight = solid_S(self.fft_freq[1:], T_tra)
                dos_gas = group['DOS 2PT']['translation'][:, 0]
                dos_solid = group['DOS 2PT']['translation'][:, 1]
                # 奈奎斯特频率贡献减半
                dos_gas[-1] *= 0.5
                dos_solid[-1] *= 0.5
                # gas-like
                group['Entropy']['translation'][0] += 0.5 * \
                    dos_gas[0]*gas_weight*delta_freq  # 0频
                group['Entropy']['translation'][0] += np.trapz(
                    y=dos_gas[1:]*gas_weight, x=self.fft_freq[1:])  # 其他
                # solid-like
                group['Entropy']['translation'][1] += np.trapz(
                    y=dos_solid[1:]*solid_weight, x=self.fft_freq[1:])  # 其他
            else:
                # 多原子分子流体，包括平动、转动、振动，含线性情况
                group['Entropy']['translation'] = [0., 0.]
                group['Entropy']['rotation'] = [0., 0.]
                group['Entropy']['vibration'] = [0., 0.]
                # 通用参数
                n_mols = group['n_mols']
                masses = group['mass']
                symmetry = group['rotation_symmetry']
                T_tra, T_omega, T_vib = group['Temperature'][0:3]
                character_Temperature = group['rot temperature']
                f_tra, _, f_omega = group['fluidicity']
                Delta_tra, _, Delta_omega = group['Delta']
                volume = group['volume']
                delta_freq = self.fft_freq[1] - self.fft_freq[0]

                # Translation
                gas_weight = gas_S_trans(
                    n_mols, masses, T_tra, volume, f_tra, Delta_tra)
                solid_weight = solid_S(self.fft_freq[1:], T_tra)
                dos_gas = group['DOS 2PT']['translation'][:, 0]
                dos_solid = group['DOS 2PT']['translation'][:, 1]
                dos_gas[-1] *= 0.5
                dos_solid[-1] *= 0.5
                # gas-like
                group['Entropy']['translation'][0] += 0.5 * \
                    dos_gas[0]*gas_weight*delta_freq  # 0频
                group['Entropy']['translation'][0] += np.trapz(
                    y=dos_gas[1:]*gas_weight, x=self.fft_freq[1:])  # 其他
                # solid-like
                group['Entropy']['translation'][1] += np.trapz(
                    y=dos_solid[1:]*solid_weight, x=self.fft_freq[1:])  # 其他

                # Rotation
                gas_weight = gas_S_rot(
                    symmetry, character_Temperature, T_omega, group['isLiner'])
                solid_weight = solid_S(self.fft_freq[1:], T_omega)
                dos_gas = group['DOS 2PT']['omega'][:, 0]
                dos_solid = group['DOS 2PT']['omega'][:, 1]
                dos_gas[-1] *= 0.5
                dos_solid[-1] *= 0.5
                # gas-like
                group['Entropy']['rotation'][0] += 0.5 * \
                    dos_gas[0]*gas_weight*delta_freq  # 0频
                group['Entropy']['rotation'][0] += np.trapz(
                    y=dos_gas[1:]*gas_weight, x=self.fft_freq[1:])  # 其他
                # solid-like
                group['Entropy']['rotation'][1] += np.trapz(
                    y=dos_solid[1:]*solid_weight, x=self.fft_freq[1:])  # 其他

                # Vibration
                if T_vib > 0:
                    # 存在此自由度才计算
                    solid_weight = solid_S(self.fft_freq[1:], T_vib)
                    dos_solid = group['DOS 2PT']['vibration'][:, 0]
                    dos_solid[-1] *= 0.5
                    # solid-like
                    group['Entropy']['vibration'][1] += solid_S0(
                        delta_freq, dos_solid[0], T_vib)
                    group['Entropy']['vibration'][1] += np.trapz(
                        # 其他
                        y=dos_solid[1:]*solid_weight, x=self.fft_freq[1:])
                else:
                    group['Entropy']['vibration'] = [0., 0.]

        for group in self.grps_info.Groups.values():
            print(f"Entropy of group: {group['name']}")
            for types in ['translation', 'rotation', 'vibration']:
                group['Entropy'][types] = [8.3138462 *
                                           s for s in group['Entropy'][types]]
                print(f"    Entropy of {types}: {group['Entropy'][types][0] + group['Entropy'][types][1]:6.4f} J/mol/K")

    def save_data(self):
        '''
        将vacf,dos保存为数据文件 
        '''
        if self.outfiles['vacf'] != None:
            vacf_df = pd.DataFrame(columns=['time/(ps)'], data=self.grps_info.dt * np.linspace(
                0, len(self.universe.trajectory), len(self.universe.trajectory)))
            for group in self.grps_info.Groups.values():
                df_temp = pd.DataFrame(columns=[f'Tra[{group["name"]}]', f'Rot[{group["name"]}]', f'Vib[{group["name"]}]', f'Ang[{group["name"]}]', f'Tot[{group["name"]}]'], data=group['vacf'])
                vacf_df = pd.concat([vacf_df, df_temp], axis=1)
            vacf_df.to_csv(
                self.outfiles['vacf'], index=False, float_format='%15.4f', sep='\t', mode='w')
            print("The VACF file have been generated!")

        if self.outfiles['dos'] != None:
            dos_df = pd.DataFrame(columns=['freq/(cm-1)'], data=self.fft_freq)
            for group in self.grps_info.Groups.values():
                df_temp = pd.DataFrame(columns=[f'Tra(GAS)[{group["name"]}]', f'Tra(SOLI)[{group["name"]}]', f'Rot(GAS)[{group["name"]}]', f'Rot(SOLI)[{group["name"]}]', f'Vib[{group["name"]}]', f'Ang(GAS)[{group["name"]}]', f'Ang(SOLI)[{group["name"]}]'], data=np.hstack([group['DOS 2PT']['translation'], group['DOS 2PT']['rotation'], group['DOS 2PT']['vibration'], group['DOS 2PT']['omega']]))
                dos_df = pd.concat([dos_df, df_temp], axis=1)
            dos_df.to_csv(self.outfiles['dos'], index=False,
                          float_format='%15.4f', sep='\t', mode='w')
            print("The power specttrum file have been generated!")

        if self.outfiles['report'] != None:
            report_df = pd.DataFrame(columns=['property'], data=[
                                     'Moleculers', 'Atoms', 'DOF', 'Temperature/(K)', 'Volume/(A3)', 'Sq/(J/mol_K)', 'Fluidicity', 'Diffusion/(1e-5 cm2/s)'])
            for group in self.grps_info.Groups.values():
                data_tra = np.array([group['n_mols'], group['n_atoms'], np.trapz(y=group['DOS'][:, 0], x=self.fft_freq), group['Temperature'][0],
                                    group['volume'], group['Entropy']['translation'][0]+group['Entropy']['translation'][1], group['fluidicity'][0], group['diffusion'][0]*1e5])
                data_rot = np.array([group['n_mols'], group['n_atoms'], np.trapz(y=group['DOS'][:, 3], x=self.fft_freq), group['Temperature'][3],
                                    group['volume'], group['Entropy']['rotation'][0]+group['Entropy']['rotation'][1], group['fluidicity'][2], group['diffusion'][3]*1e5])
                data_vib = np.array([group['n_mols'], group['n_atoms'], np.trapz(y=group['DOS'][:, 2], x=self.fft_freq), group['Temperature']
                                    [2], group['volume'], group['Entropy']['vibration'][0]+group['Entropy']['vibration'][1], 0, group['diffusion'][2]*1e5])
                data_tot = data_tra + data_rot + data_vib
                data_tot[[0, 1, 3, 4, 6, 7]] = np.array(
                    [group['n_mols'], group['n_atoms'], group['Temperature'][4], group['volume'], 0, 0])

                datas = np.vstack([data_tra, data_rot, data_vib, data_tot]).T
                df_temp = pd.DataFrame(columns=[f'Tra[{group['name']}]', f'Rot[{group["name"]}]', f'Vib[{group["name"]}]', f'Tot[{group["name"]}]'], data=datas)
                report_df = pd.concat([report_df, df_temp], axis=1)

            report_df.to_csv(
                self.outfiles['report'], index=False, float_format='%9.4f', sep='\t', mode='w')
            print("The report file have been generated!")
