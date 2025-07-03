import numpy as np


def rotational_I(coordinates, masses):
    '''
    输入:coorinates:(n_frames, n_mols, mol_atoms, 3),即输入一组的全部坐标,按分子拆分。
        masses:一个分子的质量,形状应当为(mol_atoms,)
    输出:转动惯量(n_frame, n_mols, 3, 3)
    '''
    # 质心
    center_of_mass = np.sum(
        masses[None, None, :, None] * coordinates, axis=2) / np.sum(masses)
    # 相对于质心的内部笛卡尔坐标
    relative_coords = coordinates - center_of_mass[:, :, None, :]

    # 转动惯量张量
    I_xx = np.sum(masses[None, None, :]*(relative_coords[:,
                  :, :, 1]**2 + relative_coords[:, :, :, 2]**2), axis=2)
    I_yy = np.sum(masses[None, None, :]*(relative_coords[:,
                  :, :, 0]**2 + relative_coords[:, :, :, 2]**2), axis=2)
    I_zz = np.sum(masses[None, None, :]*(relative_coords[:,
                  :, :, 0]**2 + relative_coords[:, :, :, 1]**2), axis=2)

    I_xy = -np.sum(masses[None, None, :]*relative_coords[:,
                   :, :, 0]*relative_coords[:, :, :, 1], axis=2)
    I_xz = -np.sum(masses[None, None, :]*relative_coords[:,
                   :, :, 0]*relative_coords[:, :, :, 2], axis=2)
    I_yz = -np.sum(masses[None, None, :]*relative_coords[:,
                   :, :, 1]*relative_coords[:, :, :, 2], axis=2)

    # 组合每帧每个分子的张量
    inertia_tensors = np.zeros(
        (coordinates.shape[0], coordinates.shape[1], 3, 3), dtype="float32")
    inertia_tensors[:, :, 0, 0] = I_xx
    inertia_tensors[:, :, 1, 1] = I_yy
    inertia_tensors[:, :, 2, 2] = I_zz
    inertia_tensors[:, :, 0, 1] = inertia_tensors[:, :, 1, 0] = I_xy
    inertia_tensors[:, :, 0, 2] = inertia_tensors[:, :, 2, 0] = I_xz
    inertia_tensors[:, :, 1, 2] = inertia_tensors[:, :, 2, 1] = I_yz

    return inertia_tensors


def angular_M(coordinates, velocities, masses):
    '''
    输入:coorinates:(n_frames, n_mols, mol_atoms, 3),即输入一组的全部坐标,按分子拆分。
        velocites:(n_frames, n_mols, mol_atoms, 3),即输入一组的全部速度,按分子拆分。
        masses:一个分子的质量,形状应当为(mol_atoms,)
    返回一个角动量矩阵:(n_frames, n_mols, 3)
    '''
    # 质心
    center_of_mass = np.sum(
        masses[None, None, :, None] * coordinates, axis=2) / np.sum(masses)
    # 相对于质心的内部笛卡尔坐标
    relative_coords = coordinates - center_of_mass[:, :, None, :]

    angular_momentum = np.sum(
        masses[None, None, :, None] * np.cross(relative_coords, velocities), axis=2)

    return angular_momentum



def calcu_v_rot(coordinates, velocities, masses, isLiner: bool = False):
    '''
    计算转动速度, 先计算角速度,再与相对坐标求外积得到旋转线速度, 总速度扣除平动速度和旋转的线速度得到振动内速度
    输入:coorinates:(n_frames, n_mols, mol_atoms, 3),即输入一组的全部坐标,按分子拆分。
        velocites:(n_frames, n_mols, mol_atoms, 3),即输入一组的全部速度,按分子拆分。
        masses:一个分子的质量,形状应当为(mol_atoms,)
    返回一个角速度矩阵:(n_frames, n_mols, 3), 一个线速度矩阵(n_frames, n_mols, mol_atoms, 3)
    '''
    # 质心
    center_of_mass = np.sum(
        masses[None, None, :, None] * coordinates, axis=2) / np.sum(masses)
    # 相对于质心的内部笛卡尔坐标
    relative_coords = coordinates - center_of_mass[:, :, None, :]

    # 计算每一帧的每个分子的惯量张量
    inertia_tensors = rotational_I(coordinates, masses)
    # 计算每一帧的每个分子的角动量
    angular_momenta = angular_M(coordinates, velocities, masses)
    # 转动惯量对角化，求特征值和特征向量，对称矩阵使用eigh可加速一倍，并不受奇异矩阵影响
    # 但此处只能处理惯量张量满秩的情况，对于线性分子，其转动惯量特征值中有一个是0，无法直接除
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensors)
    # 角动量分量
    if isLiner == True:
        # 对于线性分子，其转动惯量特征值中有一个是0，无法直接除
        # 这里将其取为0，其他元素取倒数
        eigenvalues[:, :, 1:] = 1.0 / eigenvalues[:, :, 1:]
        eigenvalues[:, :, 0] = 0
        omega_p = np.matmul((eigenvectors*eigenvalues[:, :, None, :]).transpose(
            0, 1, 3, 2), angular_momenta[..., None]).squeeze(-1)
        eigenvalues[:, :, 1:] = 1.0 / eigenvalues[:, :, 1:]
    else:
        # 非线性分子之间求分量即可
        omega_p = np.matmul((eigenvectors/eigenvalues[:, :, None, :]).transpose(
            0, 1, 3, 2), angular_momenta[..., None]).squeeze(-1)
    omega = np.matmul(eigenvectors, omega_p[..., None]).squeeze(-1)
    v_rot = np.cross(omega[:, :, None, :], relative_coords)

    # 计算惯量加权的角速度
    omgea_I = np.matmul(
        eigenvectors*np.sqrt(eigenvalues[:, :, None, :]), omega_p[..., None]).squeeze(-1)

    # 计算特征温度
    # rot_T = h^2*Na*1e23/(8pi^2*kB*eigenvalues)
    kb = 1.380658e-23  # 玻尔兹曼常数
    Na = 6.0221367e23  # 阿伏伽德罗常数
    h = 6.62606896e-34  # 普朗克常数
    rot_T = np.zeros((3))
    if isLiner == True:
        rot_T[1:] = (h**2 * Na * 1e23) / (8 * np.pi**2 * kb *
                                          eigenvalues[:, :, 1:]).mean(axis=0).mean(axis=0)
    else:
        rot_T = (h**2 * Na * 1e23) / (8 * np.pi**2 * kb *
                                      eigenvalues).mean(axis=0).mean(axis=0)

    return v_rot, omega, omgea_I, rot_T


def calcu_v_trans(velocities, masses):
    '''
    计算平动速度, 求得质心速度即可
    velocites:(n_frames, n_mols, mol_atoms, 3),即输入一组的全部速度,按分子拆分。
    '''
    v_center = np.sum(masses[None, None, :, None] *
                      velocities, axis=2) / np.sum(masses)

    v_tra = np.tile(v_center[:, :, None, :], (1, 1,  velocities.shape[2], 1))
    return v_tra.astype("float32")


def kieticE_aver(velocities, masses):
    '''
    输入一个速度矩阵(n_frames*n_atoms*3)一个质量向量(n_atoms,)
    输出所有帧的每个原子的平均动能(n_atoms)
    单位上质量为g/mol, 速度单位A/ps=100m/s
    返回平均动能
    '''
    KE = 10*0.5*np.sum(np.square(velocities), axis=2) * masses[None, :]
    KE_aver = np.mean(KE, axis=0)
    return KE_aver.astype("float32")
