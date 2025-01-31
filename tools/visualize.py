import numpy as np
import plotly.graph_objects as go

# 示例数据（请替换为你的实际数据）
N = 10
atom_types = np.random.randint(1, 4, size=N)  # 假设有3种原子类型
lattice_tensor = np.array([[5, 0, 0],
                           [0, 5, 0],
                           [0, 0, 5]])
atom_coords = np.random.rand(N, 3) * 5  # 在晶胞内随机分布
atom_forces = (np.random.rand(N, 3) - 0.5) * 1  # 随机力向量

# 定义原子类型对应的颜色
type_colors = {1: 'red', 2: 'green', 3: 'blue'}

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