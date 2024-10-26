# 2PT-PYTHON

2PT-PYTHON 是通过 Two-Phase Thermodynamics Model 计算绝对熵的一个 Python 实现, 除计算纯物质的熵值之外, 还基于对分子大小的估计来计算混合物的熵值, 改善了完全基于摩尔分数计算不同组分体积带来的误差。

## 安装
2PT-PYTHON 对Python versions = 3.10.14 和 Python versions = 3.12.7 进行了测试, 其他版本未进行测试, 请自行测试。附带的测试文件均在Gromacs 2023.2版本下产生。

```shell
git clone https://github.com/zdworld/2pt-python.git
cd 2pt-python
pip install -r requirements.txt
```

## 使用方法
2PT 方法可以从约 20 ps 的凝聚相轨迹或者 200 ps 的气相轨迹中获取较为准确的绝对熵, 轨迹应当通过NVT系综产生, 若通过NPT系综产生则按最后一帧的box尺寸计算。
保存的频率应当足够高, 以记录全部的振动模式, 通常来说 4 fs 的采样间隔就足够(最大频率约4100cm^-1, 包含了绝大多数振动模式)。为了结果的可靠性至少采样 5000 frames, 并检查自相关函数是否充分衰减。
### 轨迹预处理
分子动力学轨迹应当预先处理周期性边界并移除质心的平动与转动:
```shell
echo 0 | gmx trjconv -f prod.trr -s prod.tpr -o prod.pbc.trr -pbc mol
echo -e "0\n0\n" | gmx trjconv -f "prod.pbc.trr" -s prod.tpr -o prod.fit.trr -fit rot+trans
```

### 配置文件
以一份40 ps, 10001 frames$的opc3水盒子的轨迹为例(由`./gmxTraj/opc3.tpr`产生):
程序读取一份`.yaml`格式的配置文件，内容如下:

```yaml
Input:
  topology: "./gmxTraj/opc3.tpr"
  trajectory: "./gmxTraj/opc3.trr"
  Temperature: 298.15

Output:
  vacf: "./test/w_vacf.csv"
  dos: "./test/w_dos.csv"
  report: "./test/w_thermo.csv"

Groups:
  group1:
    name: "water1"
    selection: "same residue as name OW and prop z >= 20"
    density: 33
    constraints: 3
    rotation_symmetry: 2
    isLiner: false
    isCalculated: true
  group2:
    name: "water2"
    selection: "same residue as name OW and prop z < 20"
    density: 33
    constraints: 3
    rotation_symmetry: 2
    isLiner: false
    isCalculated: false
```

配置文件分为三个部分:
- `Input`: 输入文件, 包括拓扑文件路径和轨迹文件路径, 以及体系的平均温度。
- `Output`: 输出文件, vacf为质量加权的速度自相关函数, dos为态密度，report为计算结果, 若为空则不输出。
- `Groups`: 分组信息, 每个组(group1, group2...)为一个字典, 每个组必须由相同的分子构成, 且不同组之间的交集为空, 所有组的并集为全部原子。包含下列内容:
  - `name`: 组名, 会输出至报告中。
  - `selection`: 选择语句, 使用[MDAnalysis的选区语法](https://docs.mdanalysis.org/stable/documentation_pages/selections.html), 由于速度的旋转分量以分子为单位计算, 此处得到的选区总会包含这些被选中的原子所在的残基。对单个分子内具有不同残基的大分子若没有正确选中整个分子可导致错误的结果。
  - `density`: 组内分子的数密度, 用于估计摩尔体积, 单位为 nm^-1, 计算时会将全部数密度乘以系数`K_deltaV`, 如果`K_deltaV`偏离1过多则表明数密度的估计是严重偏差的, 这对纯物质没有影响但对混合物有很大的计算偏差。
  - `constraints`: 此组内每个分子被约束的自由度, 刚性水为3, 模拟中约束了`h-bonds`也应当记录在此处。
  - `rotation_symmetry`: 旋转对称数, 根据分子的点群确定, 参考[Molecular symmetry, rotational entropy, and elevated melting points](https://doi.org/10.1021/ie990588m)。
  - `isLiner`: 是否为线性分子, 线性分子只有两个转动主轴, 具有不同的转动配分函数。
  - `isCalculated`: 是否计算此组, 若为`false`则跳过此组, 但其密度数据会用于计算其他组的体积。

示例的配置文件中将全部原子分成两组:
- `water1`: `OW` 的 `z` 坐标大于等于 20 $\AA$ 的水分子, 计算其熵。
- `water2`: `OW` 的 `z` 坐标小于 20 $\AA$ 的水分子, 不计算熵。

### 运行
```shell
python3 2PT.py -c config.yaml
```
根据体系大小耗时不同, 相关信息会输出在终端, 程序消耗巨量内存, 作为参考test中的丙酮盒子(5000 atoms, 100001 frames)速度分解时消耗内存4420 MiB, 但峰值内存消耗是其3-4倍。此外要求CPU至少4核。

若体系为混合物, 体系的总熵应当减去混合熵:

$$
S_{\text{total}} = \sum_{i=1}^{n} x_i S_i -k\sum_{i=1}^{n} x_i \ln x_i
$$

其中 $x_i$ 为第 $i$ 组的摩尔分数, $S_i$ 为第 $i$ 组的熵值。


