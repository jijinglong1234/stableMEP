"""Module to generate networkx graphs."""
import json

"""Implementation based on the template of ALIGNN."""
from multiprocessing.context import ForkContext
from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
from jarvis.db.figshare import data as jdata, get_request_data

# from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
from omegaconf import ValueNode
from torch.utils.data import Dataset
import itertools
from tools.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)
# import matformer.algorithm as algorithm
import os
import pickle


try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass

# the way of diffCSP
class CrystDataset(Dataset):
    def __init__(self, name: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode,
                 use_pos_index: ValueNode, dataset_name:ValueNode, id_list_name:List,max_atoms:ValueNode, **kwargs):
        super().__init__()

        self.name = name # 'Formation energy train'
        self.prop = prop  # 'energy_per_atom'  'e_form'
        self.niggli = niggli # True
        self.primitive = primitive # False
        self.graph_method = graph_method # 'crystalnn'
        self.lattice_scale_method = lattice_scale_method  # 'scale_length'
        self.use_space_group = use_space_group # False
        self.use_pos_index = use_pos_index # False
        self.tolerance = tolerance # 0.1

        self.dataset_name = dataset_name
        self.preprocess(save_path, preprocess_workers, prop, id_list_name)   # 'preprocess_workers': 30, 'save_path': '/home/huangjiao/codes/diffcsp//data/carbon_24/train_ori.pt'

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        for data in self.cached_data:
            if len(data['graph_arrays'][0]) > max_atoms:
                self.cached_data.remove(data)
        # atom_counts = [len(data['graph_arrays'][0]) for data in self.cached_data]
        # from collections import Counter
        # distribution = Counter(atom_counts)
        # sorted_distribution = sorted(distribution.items())
        self.lattice_scaler = None
        self.scaler = None


    def preprocess(self, save_path, preprocess_workers, prop, id_list_name):

        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            try:
                data = jdata(self.dataset_name)  #'/{enviroment_name}/lib/python3.*/site-packages/jarvis/db/megnet.json.zip'
            except ValueError as e:
                #  'Check DB name options.'
                if str(e) == 'Check DB name options.':
                    if self.dataset_name == 'mp_trj':
                        # data = get_request_data('MPtrj_2022.9_full.json',
                        #                         'https://figshare.com/ndownloader/files/41619375')
                        js_tag = 'MPtrj_2022_9_full.json'
                        path = str(os.path.join(
                            '/home/huangjiao/anaconda3/envs/diffcsp2/lib/python3.8/site-packages/jarvis/db', js_tag))
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data_list = []
                        for k,v in data.items():
                            for k1, v1 in v.items():
                                v1['mptrj_id'] = k1
                                data_list.append(v1)
                        data = data_list
                    elif self.dataset_name == 'm3gnet_mpf':
                        js_tag = 'm3gnet_mpf.json'
                        path = str(os.path.join(
                            '/home/huangjiao/anaconda3/envs/diffcsp2/lib/python3.8/site-packages/jarvis/db', js_tag))
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        for i in range(len(data)):
                            material_dict = data[i]
                            energy = material_dict['energy']
                            e_form = energy/len(material_dict['force'])
                            data[i]['e_form'] = e_form
                else:
                    # raise original err原始错误
                    raise

            with open(id_list_name, 'rb') as f:
                id_list = pickle.load(f)

            # data = data[id_list]
            selected_data = [data[i] for i in id_list]

            # one_std_list = []
            # two_std_list = []
            # three_std_list = []
            # more_std_list = []

            # means = -1.6509
            # stds = 1.0713
            # for i in range(len(selected_data)):
            #     material = data[i]
            #     if abs(material['ef_per_atom'] - means) < abs(stds):
            #         one_std_list.append(i)
            #     elif abs(material['ef_per_atom'] - means) < 2 * abs(stds):
            #         two_std_list.append(i)
            #     elif abs(material['ef_per_atom'] - means) < 3 * abs(stds):
            #         three_std_list.append(i)
            #     else:
            #         more_std_list.append(i)

            cached_data = preprocess(
            selected_data,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance,
            data_type=self.dataset_name)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"

# pyg dataset
# def preprocess(self, save_path, preprocess_workers, prop, id_list_name):
class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False,
        id_tag="jid",
        neighbor_strategy="",
        lineControl=True,
        mean_train=None,
        std_train=None,
    ):
        """Pytorch Dataset for atomistic graphs.

                `df`: pandas dataframe from e.g. jarvis.db.figshare.data
                `graphs`: DGLGraph representations corresponding to rows in `df`
                `target`: key for label column in `df`
                """
        self.df = df
        self.graphs = graphs
        self.target = target
        self.line_graph = line_graph

        self.ids = self.df[id_tag]
        self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )

        # d = jdata(dataset)
        d = jdata('megnet')
        dat = []
        all_targets = []
        for i in d:
            # if isinstance(i[target], list):  # multioutput target
            #     all_targets.append(torch.tensor(i[target]))
            #     dat.append(i)
            #
            # elif (
            #         i[target] is not None
            #         and i[target] != "na"
            #         and not math.isnan(i[target])
            # ):
            #     if target_multiplication_factor is not None:
            #         i[target] = i[target] * target_multiplication_factor
            #     if classification_threshold is not None:
            #         if i[target] <= classification_threshold:
            #             i[target] = 0
            #         elif i[target] > classification_threshold:
            #             i[target] = 1
            #         else:
            #             raise ValueError(
            #                 "Check classification data type.",
            #                 i[target],
            #                 type(i[target]),
            #             )
            dat.append(i)
            all_targets.append(i[target])
        with open(id_list_name, 'rb') as f:
            id_list = pickle.load(f)
        selected_data = [dat[i] for i in id_list]
        if classification_threshold is None:
            try:
                from sklearn.metrics import mean_absolute_error

                print("MAX val:", max(all_targets))
                print("MIN val:", min(all_targets))
                print("MAD:", mean_absolute_deviation(all_targets))
                try:
                    f = open(os.path.join(output_dir, "mad"), "w")
                    line = "MAX val:" + str(max(all_targets)) + "\n"
                    line += "MIN val:" + str(min(all_targets)) + "\n"
                    line += (
                            "MAD val:"
                            + str(mean_absolute_deviation(all_targets))
                            + "\n"
                    )
                    f.write(line)
                    f.close()
                except Exception as exp:
                    print("Cannot write mad", exp)
                    pass
                # Random model precited value
                x_bar = np.mean(np.array([i[target] for i in dataset_train]))
                baseline_mae = mean_absolute_error(
                    np.array([i[target] for i in dataset_test]),
                    np.array([x_bar for i in dataset_test]),
                )
                print("Baseline MAE:", baseline_mae)
            except Exception as exp:
                print("Data error", exp)
                pass

        df = pd.DataFrame(selected_data)
        vals = df[target].values
        if target == "shear modulus" or target == "bulk modulus":
            val_list = [vals[i].item() for i in range(len(vals))]
            vals = val_list
        output_dir = "./saved_data/" + tmp_name + "test_graph_angle.pkl"  # for fast test use
        print("data range", np.max(vals), np.min(vals))
        print(output_dir)

        graphs = load_pyg_graphs(
            df,
            name=name,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )




        print("mean %f std %f"%(self.labels.mean(), self.labels.std()))
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            print("normalize using training mean %f and std %f" % (mean_train, std_train))

        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f

        self.prepare_batch = prepare_pyg_batch
        if line_graph:
            self.prepare_batch = prepare_pyg_line_graph_batch
            print("building line graphs")
            if lineControl == False:
                self.line_graphs = []
                self.graphs = []
                for g in tqdm(graphs):
                    linegraph_trans = LineGraph(force_directed=True)
                    g_new = Data()
                    g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
                    try:
                        lg = linegraph_trans(g)
                    except Exception as exp:
                        print(g.x, g.edge_attr, exp)
                        pass
                    lg.edge_attr = pyg_compute_bond_cosines(lg) # old cosine emb
                    # lg.edge_attr = pyg_compute_bond_angle(lg)
                    self.graphs.append(g_new)
                    self.line_graphs.append(lg)
            else:
                if neighbor_strategy == "pairwise-k-nearest":
                    self.graphs = []
                    labels = []
                    idx_t = 0
                    filter_out = 0
                    max_size = 0
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        if g.x.size(0) > max_size:
                            max_size = g.x.size(0)
                        if g.x.size(0) < 200:
                            self.graphs.append(g)
                            labels.append(self.labels[idx_t])
                        else:
                            filter_out += 1
                        idx_t += 1
                    print("filter out %d samples because of exceeding threshold of 200 for nn based method" % filter_out)
                    print("dataset max atom number %d" % max_size)
                    self.line_graphs = self.graphs
                    self.labels = labels
                    self.labels = torch.tensor(self.labels).type(
                                    torch.get_default_dtype()
                                )
                else:
                    self.graphs = []
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        self.graphs.append(g)
                    self.line_graphs = self.graphs


        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            if 'node_pos' in g:
                node_pos = g.node_pos
                return g, self.line_graphs[idx], self.ids[idx], label, node_pos
            return g, self.line_graphs[idx], label, label

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattice, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.tensor(labels)

    @staticmethod
    def collate_line_graph_knowledge(
            samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        value_num = len(list(map(list, zip(*samples))))
        if value_num == 5:
            graphs, line_graphs, lattice, labels, node_pos = map(list, zip(*samples))
        elif value_num == 4:
            graphs, line_graphs, lattice, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        if value_num == 4:
            if len(labels[0].size()) > 0:
                return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.stack(labels)
            else:
                return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.tensor(labels)
        elif value_num == 5:
            if len(labels[0].size()) > 0:
                return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.stack(labels),node_pos
            else:
                # return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.tensor(labels),node_pos
                return batched_graph, batched_line_graph, lattice, torch.tensor(labels), node_pos

class VisualizeDataset(Dataset):
    def __init__(self, name: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode,
                 use_pos_index: ValueNode, dataset_name:ValueNode,mpid,grid_size,sigma,max_atoms:ValueNode, **kwargs):
        super().__init__()

        self.name = name # 'Formation energy train'
        self.prop = prop  # 'energy_per_atom'  'e_form'
        self.niggli = niggli # True
        self.primitive = primitive # False
        self.graph_method = graph_method # 'crystalnn'
        self.lattice_scale_method = lattice_scale_method  # 'scale_length'
        self.use_space_group = use_space_group # False
        self.use_pos_index = use_pos_index # False
        self.tolerance = tolerance # 0.1

        self.dataset_name = dataset_name
        self.preprocess(save_path, preprocess_workers, prop, mpid,grid_size,sigma)   # 'preprocess_workers': 30, 'save_path': '/home/huangjiao/codes/diffcsp//data/carbon_24/train_ori.pt'

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        for data in self.cached_data:
            if len(data['graph_arrays'][0]) > max_atoms:
                self.cached_data.remove(data)
        self.lattice_scaler = None
        self.scaler = None


    def preprocess(self, save_path, preprocess_workers, prop,mpid,grid_size,sigma,mu=0):

        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            try:
                data = jdata(self.dataset_name)  #'/{enviroment_name}/lib/python3.*/site-packages/jarvis/db/megnet.json.zip'
            except ValueError as e:
                #  'Check DB name options.'
                if str(e) == 'Check DB name options.':
                    if self.dataset_name == 'mp_trj':
                        # data = get_request_data('MPtrj_2022.9_full.json',
                        #                         'https://figshare.com/ndownloader/files/41619375')
                        js_tag = 'MPtrj_2022_9_full.json'
                        path = str(os.path.join(
                            '/home/huangjiao/anaconda3/envs/diffcsp2/lib/python3.8/site-packages/jarvis/db', js_tag))
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data_list = []
                        for k,v in data.items():
                            for k1, v1 in v.items():
                                v1['mptrj_id'] = k1
                                data_list.append(v1)
                        data = data_list
                    elif self.dataset_name == 'm3gnet_mpf':
                        js_tag = 'm3gnet_mpf.json'
                        path = str(os.path.join(
                            '/home/huangjiao/anaconda3/envs/diffcsp2/lib/python3.8/site-packages/jarvis/db', js_tag))
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                else:
                    # raise original err
                    raise

            for mpdata in data:
                if mpdata['id'] == mpid:
                    chosen_material = mpdata
                    break
            chosen_material = preprocess(
                [chosen_material],
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
                data_type=self.dataset_name)

            chosen_material = chosen_material[0]

            X = chosen_material['graph_arrays'][0]
            D1 = np.random.normal(mu, sigma, X.shape)
            D2 = np.random.normal(mu, sigma, X.shape)
            # 定义网格的大小
            # grid_size = 20  # 生成20x20的网格
            i_values = np.linspace(-1, 1, grid_size)
            j_values = np.linspace(-1, 1, grid_size)

            self.i_values = i_values
            self.j_values = j_values

            selected_data = []

            for i_idx, i in enumerate(i_values):
                for j_idx, j in enumerate(j_values):
                    # 计算 X'(i, j) = X + i * D1 + j * D2
                    X_prime = X + i * D1 + j * D2
                    x_new = dict()
                    for k, v in chosen_material.items():
                        x_new[k] = v
                    x_new['graph_arrays'] = (X_prime, *chosen_material['graph_arrays'][1:])
                    selected_data.append(x_new)

            # cached_data = preprocess(
            # selected_data,
            # preprocess_workers,
            # niggli=self.niggli,
            # primitive=self.primitive,
            # graph_method=self.graph_method,
            # prop_list=[prop],
            # use_space_group=self.use_space_group,
            # tol=self.tolerance,
            # data_type=self.dataset_name)

            self.cached_data = selected_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"VisualizeDataset({self.name=}, {self.path=})"



def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))
            
    return edges



def pair_nearest_neighbor_edges(
        atoms=None,
        pair_wise_distances=6,
        use_lattice=False,
        use_angle=False,
):
    """Construct pairwise k-fully connected edge list."""
    smallest = pair_wise_distances
    lattice_list = torch.as_tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]).float()

    lattice = torch.as_tensor(atoms.lattice_mat).float()
    pos = torch.as_tensor(atoms.cart_coords)
    atom_num = pos.size(0)
    lat = atoms.lattice
    radius_needed = min(lat.a, lat.b, lat.c) * (smallest / 2 - 1e-9)
    r_a = (np.floor(radius_needed / lat.a) + 1).astype(np.int)
    r_b = (np.floor(radius_needed / lat.b) + 1).astype(np.int)
    r_c = (np.floor(radius_needed / lat.c) + 1).astype(np.int)
    period_list = np.array([l for l in itertools.product(*[list(range(-r_a, r_a + 1)), list(range(-r_b, r_b + 1)), list(range(-r_c, r_c + 1))])])
    period_list = torch.as_tensor(period_list).float()
    n_cells = period_list.size(0)
    offset = torch.matmul(period_list, lattice).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offset).transpose(0, 1).contiguous()
    dist = (pos.unsqueeze(1).unsqueeze(1) - expand_pos.unsqueeze(0))  # [n, 1, 1, 3] - [1, n, n_cell, 3] -> [n, n, n_cell, 3]
    dist2, index = torch.sort(dist.norm(dim=-1), dim=-1, stable=True)
    max_value = dist2[:, :, smallest - 1]  # [n, n]
    mask = (dist.norm(dim=-1) <= max_value.unsqueeze(-1))  # [n, n, n_cell]
    shift = torch.matmul(lattice_list, lattice).repeat(atom_num, 1)
    shift_src = torch.arange(atom_num).unsqueeze(-1).repeat(1, lattice_list.size(0))
    shift_src = torch.cat([shift_src[i,:] for i in range(shift_src.size(0))])
    
    indices = torch.where(mask)
    dist_target = dist[indices]
    u, v, _ = indices
    if use_lattice:
        u = torch.cat((u, shift_src), dim=0)
        v = torch.cat((v, shift_src), dim=0)
        dist_target = torch.cat((dist_target, shift), dim=0)
        assert u.size(0) == dist_target.size(0)

    return u, v, dist_target

def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r

def build_undirected_edgedata_tensor(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    # build_undirected_edgedata

    # frac_coords = torch.tensor(atoms.frac_coords).requires_grad_()
    # frac_coords = frac_coords.to(torch.float32)
    # u, v, r = [], [], []
    # for (src_id, dst_id), images in edges.items():
    #     for dst_image in images:
    #         # fractional coordinate for periodic image of dst
    #         dst_coord = frac_coords[dst_id] + torch.tensor(dst_image).to(torch.float32)
    #         src_coord = frac_coords[src_id]
    #         # cartesian displacement vector pointing from src -> dst
    #         lattice = torch.tensor(atoms.lattice.lattice()).to(torch.float32)
    #         d = dst_coord - src_coord
    #         d = torch.matmul(d, lattice)
    #         for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
    #             u.append(uu)
    #             v.append(vv)
    #             r.append(dd)
    # u = torch.tensor(u)
    # v = torch.tensor(v)
    # r = torch.stack(r, dim=0)

    lattice = torch.tensor(atoms.lattice.lattice()).to(torch.float32)
    # def calculate_adjacent_distances_multi_graph_optimized(coordinates, adjacency_dict, lattice):
    # 将分数坐标转换为笛卡尔坐标
    # coordinates_cartesian1 = torch.mm(frac_coords, lattice)
    coordinates_cartesian = torch.tensor(atoms.cart_coords, dtype=torch.float32).requires_grad_()

    # # 预处理阶段：创建一个存储（i, j, shift）信息的矩阵
    triplets = []
    for (i, j), lattice_shifts in edges.items():
        for shift in lattice_shifts:
            triplets.append((i, j, *shift))

    triplets = torch.tensor(triplets, dtype=torch.int)
    i_indices = triplets[:, 0]
    j_indices = triplets[:, 1]
    shifts = triplets[:, 2:]

    # 使用广播和矩阵乘法来一次性计算所有距离
    coord_i = coordinates_cartesian[i_indices]
    coord_j = coordinates_cartesian[j_indices]
    shift_cartesian = torch.mm(shifts.float(), lattice)
    distances = (coord_j + shift_cartesian) - coord_i
    u1 = torch.cat([i_indices,j_indices])
    v1 = torch.cat([j_indices,i_indices])
    r1 = torch.cat([distances, -distances])
    # for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
    #     u1.append(i.item())
    #     v1.append(j.item())
    #     r1.append(distances[idx])
    #     u1.append(j.item())
    #     v1.append(i.item())
    #     r1.append(-distances[idx])
    #
    # u1 = torch.tensor(u1)
    # v1 = torch.tensor(v1)
    # r1 = torch.stack(r1, dim=0)

    return u1, v1, r1, coordinates_cartesian

class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0, 
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
    ):
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            # u, v, r = build_undirected_edgedata(atoms, edges)
            u, v, r, cart_coords = build_undirected_edgedata_tensor(atoms, edges)

        elif neighbor_strategy == "pairwise-k-nearest":
            u, v, r = pair_nearest_neighbor_edges(
                atoms=atoms,
                pair_wise_distances=2,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))  #"cgcnn" or "atomic number"
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()

        # g = Data(x=node_features, edge_index=edge_index, edge_attr=r)
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r, node_pos=cart_coords)
        # g = Data(x=node_features, edge_attr=r.norm(dim=-1), edge_index=torch.stack([u, v]), inf_edge_index=inf_edge_index,
        #             inf_edge_attr=inf_edge_attr)

        if compute_line_graph:
            linegraph_trans = LineGraph(force_directed=True)
            g_new = Data()
            g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
            lg = linegraph_trans(g)
            lg.edge_attr = pyg_compute_bond_cosines(lg)
            return g_new, lg
        else:
            return g

def pyg_compute_bond_cosines(lg):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def pyg_compute_bond_angle(lg):
    """Compute bond angle from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    a = (r1 * r2).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(r1, r2).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle



class PygStandardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
        """Apply standardization to atom_features."""
        h = g.x
        g.x = (h - self.mean) / self.std
        return g



def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[Data, Data, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    if len(batch) == 4:
        g, lg, lattice, t = batch
        batch = (
            (
                g.to(device),
                lg.to(device),
                lattice.to(device, non_blocking=non_blocking),
            ),
            t.to(device, non_blocking=non_blocking),
        )
    elif len(batch) == 5:
        g, lg, lattice, t, node_pos_list = batch
        batch = (
            (
                g.to(device),
                lg.to(device),
                # lattice.to(device, non_blocking=non_blocking),
                lattice,
                node_pos_list,
            ),
            t.to(device, non_blocking=non_blocking),
        )

    return batch

