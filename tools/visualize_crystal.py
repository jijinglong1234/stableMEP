import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from mpl_toolkits.mplot3d import Axes3D

# def visualize_crystal(atom_types, lattice, atom_coords, forces, frac_coords=True):
#     """
#     :param atom_types: (N,) ndarray. Atomic types.
#     :param lattice: (3, 3) ndarray. Lattice vectors.
#     :param atom_coords: (N, 3) ndarray. Atomic coordinates.
#     :param forces: (N, 3) ndarray. Force vectors on each atom.
#     """
#     # 创建一个3D绘图对象
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 绘制晶格
#     origin = np.zeros(3)
#     for i in range(3):
#         ax.quiver(*origin, *lattice[i], color='k', linewidth=2)
#
#     if frac_coords:
#         atom_coords = np.dot(atom_coords, lattice)
#
#     # 绘制原子位置和受力向量
#     for i in range(len(atom_types)):
#         # 根据原子类型设置不同的颜色或大小
#         if atom_types[i] == 1:  # 举例，假设原子类型1是红色
#             ax.scatter(*atom_coords[i], color='r', s=100)
#         elif atom_types[i] == 2:  # 假设原子类型2是蓝色
#             ax.scatter(*atom_coords[i], color='b', s=100)
#
#         # 绘制受力向量
#         ax.quiver(*atom_coords[i], *forces[i], color='g', linewidth=1)
#
#     # 设置坐标轴标签
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#
#     # 设置坐标轴范围
#     max_range = np.array([atom_coords[:, 0].max() - atom_coords[:, 0].min(),
#                           atom_coords[:, 1].max() - atom_coords[:, 1].min(),
#                           atom_coords[:, 2].max() - atom_coords[:, 2].min()]).max() / 2.0
#
#     mid_x = (atom_coords[:, 0].max() + atom_coords[:, 0].min()) * 0.5
#     mid_y = (atom_coords[:, 1].max() + atom_coords[:, 1].min()) * 0.5
#     mid_z = (atom_coords[:, 2].max() + atom_coords[:, 2].min()) * 0.5
#
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
#
#     # 显示图形
#     plt.show()


def visualize_crystal(atom_types, lattice_tensor, atom_coords, atom_forces, frac_coords=True):
    """
    :param atom_types: (N,) ndarray. Atomic types.
    :param lattice_tensor: (3, 3) ndarray. Lattice vectors.
    :param atom_coords: (N, 3) ndarray. Atomic coordinates.
    :param atom_forces: (N, 3) ndarray. Force vectors on each atom.
    :param frac_coords: True or False. If atomic coordinates are fractional.
    """

    # 定义原子类型对应的颜色
    # 获取颜色映射方案
    cmap = plt.get_cmap('tab20')  # tab20 提供20种不同的颜色

    # 为每个原子类型分配颜色
    atom_types_all = list(range(1, 101))  # 假设原子类型为1到100
    type_colors = {}

    # 对每个原子类型分配颜色
    for i, atom_type in enumerate(atom_types_all):
        type_colors[atom_type] = cmap(i % cmap.N)  # cmap.N是颜色映射方案的颜色总数

    if frac_coords:
        atom_coords = np.dot(atom_coords, lattice)

    # 创建原子散点
    scatter_atoms = go.Scatter3d(
        x=atom_coords[:, 0],
        y=atom_coords[:, 1],
        z=atom_coords[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=[type_colors[t] for t in atom_types],
            symbol='circle'
        ),
        name='Atoms'
    )

    # 创建力向量（使用箭头表示）
    # Plotly 没有内置的箭头，因此使用线条和小球来模拟箭头
    force_traces = []
    for i in range(N):
        start = atom_coords[i]
        end = start + atom_forces[i]
        # 线条
        force_traces.append(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
        # 箭头头部（小球）
        force_traces.append(go.Scatter3d(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            mode='markers',
            marker=dict(size=3, color='black'),
            showlegend=False
        ))

    # 绘制晶胞边界
    # 获取晶胞的8个顶点
    a1, a2, a3 = lattice_tensor
    vertices = np.array([
        [0, 0, 0],
        a1,
        a2,
        a3,
        a1 + a2,
        a1 + a3,
        a2 + a3,
        a1 + a2 + a3]
    )

    # 定义晶胞的12条边
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7),
        (5, 7),
        (6, 7)
    ]

    # 创建晶胞边界的线条
    lattice_lines = []
    for edge in edges:
        start, end = vertices[edge[0]], vertices[edge[1]]
        lattice_lines.append(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='grey', width=2),
            showlegend=False
        ))

    # 创建图形
    fig = go.Figure()

    # 添加晶胞边界
    for line in lattice_lines:
        fig.add_trace(line)

    # 添加原子
    fig.add_trace(scatter_atoms)

    # 添加力向量
    for trace in force_traces:
        fig.add_trace(trace)

    # 设置图形布局
    fig.update_layout(
        title='晶体结构可视化',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        legend=dict(itemsizing='constant')
    )

    # 添加图例
    # 创建自定义图例
    legend_traces = []
    for t, color in type_colors.items():
        legend_traces.append(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(size=5, color=color),
            name=f'类型 {t}'
        ))

    for trace in legend_traces:
        fig.add_trace(trace)

    # 显示图形
    fig.show()


# 示例数据
# N = 5  # 假设有5个原子
# atom_types = np.array([1, 1, 2, 2, 1])
# lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 5  # 简单立方晶格
# atom_coords = np.random.rand(N, 3) * 5
# forces = np.random.rand(N, 3) - 0.5  # 随机生成受力向量

# 示例数据（请替换为你的实际数据）
N = 10
atom_types = np.random.randint(1, 4, size=N)  # 假设有3种原子类型
lattice = np.array([[5, 0, 0],
                           [0, 5, 0],
                           [0, 0, 5]])
atom_coords = np.random.rand(N, 3) * 5  # 在晶胞内随机分布
forces = (np.random.rand(N, 3) - 0.5) * 1  # 随机力向量

# 调用可视化函数
visualize_crystal(atom_types, lattice, atom_coords, forces)
