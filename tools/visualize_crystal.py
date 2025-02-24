import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def visualize_crystal(atom_types, lattice_tensor, atom_coords, atom_forces, frac_coords=True):
    """
    :param atom_types: (N,) ndarray. Atomic types.
    :param lattice_tensor: (3, 3) ndarray. Lattice vectors.
    :param atom_coords: (N, 3) ndarray. Atomic coordinates.
    :param atom_forces: (N, 3) ndarray. Force vectors on each atom.
    :param frac_coords: True or False. If atomic coordinates are fractional.
    """
    cmap = plt.get_cmap('tab20')  
    atom_types_all = list(range(1, 101))  
    type_colors = {}

    for i, atom_type in enumerate(atom_types_all):
        type_colors[atom_type] = cmap(i % cmap.N)  

    if frac_coords:
        atom_coords = np.dot(atom_coords, lattice)

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

    force_traces = []
    for i in range(N):
        start = atom_coords[i]
        end = start + atom_forces[i]
        force_traces.append(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
        force_traces.append(go.Scatter3d(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            mode='markers',
            marker=dict(size=3, color='black'),
            showlegend=False
        ))


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


    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7),
        (5, 7),
        (6, 7)
    ]

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

    fig = go.Figure()


    for line in lattice_lines:
        fig.add_trace(line)


    fig.add_trace(scatter_atoms)


    for trace in force_traces:
        fig.add_trace(trace)

    fig.update_layout(
        title='',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        legend=dict(itemsizing='constant')
    )


    legend_traces = []
    for t, color in type_colors.items():
        legend_traces.append(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(size=5, color=color),
            name=f' {t}'
        ))

    for trace in legend_traces:
        fig.add_trace(trace)


    fig.show()



N = 10
atom_types = np.random.randint(1, 4, size=N)  # 假设有3种原子类型
lattice = np.array([[5, 0, 0],
                           [0, 5, 0],
                           [0, 0, 5]])
atom_coords = np.random.rand(N, 3) * 5  # 在晶胞内随机分布
forces = (np.random.rand(N, 3) - 0.5) * 1  # 随机力向量

visualize_crystal(atom_types, lattice, atom_coords, forces)
