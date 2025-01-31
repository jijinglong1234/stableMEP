import heapq
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools
import random
import pickle

from pymatgen.analysis.local_env import VoronoiNN, MinimumVIRENN
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from networkx.algorithms.components import is_connected

from sklearn.metrics import accuracy_score, recall_score, precision_score

from torch_scatter import scatter
from torch_scatter import segment_coo, segment_csr

from p_tqdm import p_umap

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pyxtal.symmetry import Group
from pyxtal import pyxtal

from pathos.pools import ProcessPool as Pool
# from multiprocessing import Pool
from tqdm import tqdm 
from functools import partial 

import faulthandler

from torch_geometric.data import Data, Batch

faulthandler.enable()
import time
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data, get_node_attributes, atomic_numbers_to_symbols

# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal

def refine_spacegroup(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_conventional_standard_structure()
    space_group = spga.get_space_group_number()
    crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return crystal, space_group


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.
    sym_info = {
        'anchors':anchors,
        'wyckoff_ops':matrices,
        'spacegroup':space_group
    }
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info

def build_crystal_graph(crystal, graph_method='crystalnn'):
    """
    """

    if graph_method == 'crystalnn':
        try:
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
        except:
            try:
                crystalNN_tmp = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=14)   #search_cutoff=14
                crystal_graph = StructureGraph.with_local_env_strategy(crystal, crystalNN_tmp)
            except:
                try:
                    crystalNN_tmp = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=20)
                    crystal_graph = StructureGraph.with_local_env_strategy(crystal, crystalNN_tmp)
                except:
                    try:
                        voronoiNN = VoronoiNN(cutoff=15)  # 增大 Voronoi 图的截断范围
                        crystal_graph = StructureGraph.with_local_env_strategy(crystal, voronoiNN)
                    except:
                        min_vi_nn = MinimumVIRENN()
                        crystal_graph = StructureGraph.with_local_env_strategy(crystal, min_vi_nn)

    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum('bi,bi->b', vector_a,
                                  torch.cross(vector_b, vector_c, dim=1)))


def lengths_angles_to_volume(lengths, angles):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return compute_volume(lattice)


def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix ** 2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) /
                            (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma

def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True,
    lattices = None
):
    if regularized:
        frac_coords = frac_coords % 1.
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    if regularized:
        frac_coords = frac_coords % 1.
    return frac_coords


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
    lattices=None
):
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattices, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc_wrapper(data, radius, max_num_neighbors_threshold, device):
    cart_coords = frac_to_cart_coords(
        data.frac_coords, data.lengths, data.angles, data.num_atoms)
    return radius_graph_pbc(
        cart_coords, data.lengths, data.angles, data.num_atoms, radius,
        max_num_neighbors_threshold, device)

def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res



def radius_graph_pbc(pos, lengths, angles, natoms, radius, max_num_neighbors_threshold, device, lattices=None):
    
    # device = pos.device
    batch_size = len(natoms)
    if lattices is None:
        cell = lattice_params_to_matrix_torch(lengths, angles)
    else:
        cell = lattices
    # position of the atoms
    atom_pos = pos


    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1,1)

    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1,1)
    
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1,1)
    
    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1) # N_graphs * 3
#     reps = torch.cat([rep_a1.reshape(-1,1), rep_a2.reshape(-1,1), rep_a3.reshape(-1,1)], dim=1) # N_graphs * 3
    
    unit_cell_all = []
    num_cells_all = []

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    
    unit_cell = torch.cat([_.reshape(-1,1) for _ in torch.meshgrid(cells_per_dim)], dim=-1)
    
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

#     # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    
    
    radius_real = (min_dist.min(dim=-1)[0] + 0.01)#.clamp(max=radius)
    
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)

    # print(min_dist.min(dim=-1)[0])
    
    # radius_real = radius
    
    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    if max_num_neighbors_threshold is not None:

        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)
            
    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = torch.zeros(
            natoms.shape[0] + 1, device=device, dtype=torch.long
        )
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(
    natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_real_cutoff = distance_sort[:,max_num_neighbors_threshold].reshape(-1,1).expand(-1,max_num_neighbors) + 0.01
    
    mask_distance = distance_sort < distance_real_cutoff
    
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    
    
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
#     index_sort = torch.masked_select(index_sort, mask_finite)
    index_sort = torch.masked_select(index_sort, mask_finite & mask_distance)
    
    num_neighbor_per_node = (mask_finite & mask_distance).sum(dim=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)
    

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def radius_graph_pbc_(cart_coords, lengths, angles, num_atoms,
                     radius, max_num_neighbors_threshold, device,
                     topk_per_pair=None):
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        (atom_count_sqr // num_atoms_per_image_expand)
    ).long() + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index +
            torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
        topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    )

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image
        else:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image, topk_mask

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         num_atoms, device, return_vector=False,
                         return_to_jimages=False):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(cart_coords2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector ** 2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        X = X.clone().detach().float()
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


def get_scaler_from_data_list(data_list, key):
    # targets_np = np.array([d[key] for d in data_list])
    # targets = torch.tensor(targets_np)
    targets = torch.tensor([d[key] for d in data_list])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


def process_one(row, niggli, primitive, graph_method, prop_list, use_space_group = False, tol=0.01):
    crystal_str = row['cif']
    crystal = build_crystal(
        crystal_str, niggli=niggli, primitive=primitive)
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol = tol)
        result_dict.update(sym_info)
    else:
        result_dict['spacegroup'] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
    result_dict.update({
        'mp_id': row['material_id'],
        'cif': crystal_str,
        'graph_arrays': graph_arrays
    })
    result_dict.update(properties)
    return result_dict

def process_one_jdata(jarvis_data, niggli, primitive, graph_method, prop_list, use_space_group = False, tol=0.01):
    jarvis_atoms = jarvis_data['atoms']
    lattice = jarvis_atoms['lattice_mat']
    elements = jarvis_atoms['elements']
    coords = jarvis_atoms['coords']
    crystal = Structure(lattice, elements, coords)

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    crystal = canonical_crystal
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol = tol)
        result_dict.update(sym_info)
    else:
        result_dict['spacegroup'] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: jarvis_data[k] for k in prop_list if (k in jarvis_data.keys() and k != 'atoms')}
    result_dict.update({
        'mp_id': jarvis_data['id'],
        'graph_arrays': graph_arrays
    })
    result_dict.update(properties)
    return result_dict

def process_one_mptrj(mptrj_data, niggli, primitive, graph_method, prop_list, use_space_group = False, tol=0.01):
    # jarvis_atoms = mptrj_data['atoms']
    # lattice = jarvis_atoms['lattice_mat']
    # elements = jarvis_atoms['elements']
    # coords = jarvis_atoms['coords']
    # crystal = Structure(lattice, elements, coords)
    crystal = Structure.from_dict(mptrj_data['structure'])
    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    crystal = canonical_crystal
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol = tol)
        result_dict.update(sym_info)
    else:
        result_dict['spacegroup'] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: mptrj_data[k] for k in prop_list if (k in mptrj_data.keys() and k != 'atoms')}
    result_dict.update({
        'mp_id': mptrj_data['mp_id'],
        'mptrj_id': mptrj_data['mptrj_id'],
        'graph_arrays': graph_arrays
    })
    result_dict.update(properties)
    return result_dict

def process_one_dft_3d(jarvis_data, niggli, primitive, graph_method, prop_list, use_space_group = False, tol=0.01):
    jarvis_atoms = jarvis_data['atoms']
    lattice = jarvis_atoms['lattice_mat']
    elements = jarvis_atoms['elements']
    coords = jarvis_atoms['coords']
    crystal = Structure(lattice, elements, coords)

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    crystal = canonical_crystal
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol = tol)
        result_dict.update(sym_info)
    else:
        result_dict['spacegroup'] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: jarvis_data[k] for k in prop_list if (k in jarvis_data.keys() and k != 'atoms')}
    result_dict.update({
        'jid': jarvis_data['jid'],
        'graph_arrays': graph_arrays
    })
    result_dict.update(properties)
    return result_dict

def preprocess(input_file, num_workers, niggli, primitive, graph_method,
               prop_list, use_space_group = False, tol=0.01,data_type = 'diffcsp'):
    if data_type == 'diffcsp':
        df = pd.read_csv(input_file)
        process = process_one
        data = [df.iloc[idx] for idx in range(len(df))]
    elif data_type == 'megnet':
        df = input_file
        process = process_one_jdata
        data = df
    elif data_type == 'mp_trj':
        df = input_file
        process = process_one_mptrj
        data = df
    elif data_type == 'dft_3d':
        df = input_file
        process = process_one_dft_3d
        data = df
    elif data_type == 'm3gnet_mpf':
        df = input_file
        process = process_one_jdata
        data = df
    unordered_results = p_umap(
        process,
        data,
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list[0].split(' ')] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers)

    if data_type == 'diffcsp':
        mpid_to_results = {result['mp_id']: result for result in unordered_results}
        ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
                           for idx in range(len(df))]
    elif data_type == 'megnet':
        mpid_to_results = {result['mp_id']: result for result in unordered_results}
        ordered_results = [mpid_to_results[data[idx]['id']]
                           for idx in range(len(data))]
    elif data_type == 'mp_trj':
        mpid_to_results = {result['mptrj_id']: result for result in unordered_results}
        ordered_results = [mpid_to_results[data[idx]['mptrj_id']]
                           for idx in range(len(data))]
    elif data_type == 'dft_3d':
        mpid_to_results = {result['jid']: result for result in unordered_results}
        ordered_results = [mpid_to_results[data[idx]['jid']]
                           for idx in range(len(data))]
    elif data_type == 'm3gnet_mpf':
        mpid_to_results = {result['mp_id']: result for result in unordered_results}
        ordered_results = [mpid_to_results[data[idx]['id']]
                           for idx in range(len(data))]
    return ordered_results


def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array['frac_coords']
        atom_types = crystal_array['atom_types']
        lengths = crystal_array['lengths']
        angles = crystal_array['angles']
        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            'batch_idx': batch_idx,
            'graph_arrays': graph_arrays,
        }
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(
        sorted(unordered_results, key=lambda x: x['batch_idx']))
    return ordered_results


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict['graph_arrays']
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)

        dict['scaled_lattice'] = np.concatenate([lengths, angles])


def mard(targets, preds):
    """Mean absolute relative difference."""
    assert torch.all(targets > 0.)
    return torch.mean(torch.abs(targets - preds) / targets)


def batch_accuracy_precision_recall(
    pred_edge_probs,
    edge_overlap_mask,
    num_bonds
):
    if (pred_edge_probs is None and edge_overlap_mask is None and
            num_bonds is None):
        return 0., 0., 0.
    pred_edges = pred_edge_probs.max(dim=1)[1].float()
    target_edges = edge_overlap_mask.float()

    start_idx = 0
    accuracies, precisions, recalls = [], [], []
    for num_bond in num_bonds.tolist():
        pred_edge = pred_edges.narrow(
            0, start_idx, num_bond).detach().cpu().numpy()
        target_edge = target_edges.narrow(
            0, start_idx, num_bond).detach().cpu().numpy()

        accuracies.append(accuracy_score(target_edge, pred_edge))
        precisions.append(precision_score(
            target_edge, pred_edge, average='binary'))
        recalls.append(recall_score(target_edge, pred_edge, average='binary'))

        start_idx = start_idx + num_bond

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def save_lists(lists, filenames):
    for lst, filename in zip(lists, filenames):
        with open(filename, 'wb') as f:
            pickle.dump(lst, f)


def load_lists(filenames):
    loaded_lists = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            loaded_lists.append(pickle.load(f))
    return loaded_lists


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
    period_list = np.array([l for l in itertools.product(
        *[list(range(-r_a, r_a + 1)), list(range(-r_b, r_b + 1)), list(range(-r_c, r_c + 1))])])
    period_list = torch.as_tensor(period_list).float()
    n_cells = period_list.size(0)
    offset = torch.matmul(period_list, lattice).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offset).transpose(0, 1).contiguous()
    dist = (pos.unsqueeze(1).unsqueeze(1) - expand_pos.unsqueeze(
        0))  # [n, 1, 1, 3] - [1, n, n_cell, 3] -> [n, n, n_cell, 3]
    dist2, index = torch.sort(dist.norm(dim=-1), dim=-1, stable=True)
    max_value = dist2[:, :, smallest - 1]  # [n, n]
    mask = (dist.norm(dim=-1) <= max_value.unsqueeze(-1))  # [n, n, n_cell]
    shift = torch.matmul(lattice_list, lattice).repeat(atom_num, 1)
    shift_src = torch.arange(atom_num).unsqueeze(-1).repeat(1, lattice_list.size(0))
    shift_src = torch.cat([shift_src[i, :] for i in range(shift_src.size(0))])

    indices = torch.where(mask)
    dist_target = dist[indices]
    u, v, _ = indices
    if use_lattice:
        u = torch.cat((u, shift_src), dim=0)
        v = torch.cat((v, shift_src), dim=0)
        dist_target = torch.cat((dist_target, shift), dim=0)
        assert u.size(0) == dist_target.size(0)

    return u, v, dist_target
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
    stime = time.time()
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    print("get_all_neighbors time", time.time() - stime)

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

    # stime = time.time()
    # edges = defaultdict(set)
    # for site_idx, neighborlist in enumerate(all_neighbors):
    #
    #     # sort on distance
    #     neighborlist = sorted(neighborlist, key=lambda x: x[2])
    #     distances = np.array([nbr[2] for nbr in neighborlist])
    #     ids = np.array([nbr[1] for nbr in neighborlist])
    #     images = np.array([nbr[3] for nbr in neighborlist])
    #
    #     # find the distance to the k-th nearest neighbor
    #     max_dist = distances[max_neighbors - 1]
    #     ids = ids[distances <= max_dist]
    #     images = images[distances <= max_dist]
    #     distances = distances[distances <= max_dist]
    #     for dst, image in zip(ids, images):
    #         src_id, dst_id, src_image, dst_image = canonize_edge(
    #             site_idx, dst, (0, 0, 0), tuple(image)
    #         )
    #         if use_canonize:
    #             edges[(src_id, dst_id)].add(dst_image)
    #         else:
    #             edges[(site_idx, dst)].add(tuple(image))
    #
    #     if use_lattice:
    #         edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
    #         edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
    #         edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
    #         edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
    #         edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
    #         edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))
    # print("for loop time", time.time() - stime," all_neighbors num:",len(all_neighbors))
    # edges_old = edges

    # stime = time.time()
    # edges = defaultdict(set)
    # for site_idx, neighborlist in enumerate(all_neighbors):
    #     # 找到前 max_neighbors 个最近邻居，使用 heapq.nsmallest 而不是完全排序
    #     neighborlist = heapq.nsmallest(max_neighbors, neighborlist, key=lambda x: x[2])
    #
    #     # 提取距离、id 和图像数组
    #     distances = np.array([nbr[2] for nbr in neighborlist])
    #     ids = np.array([nbr[1] for nbr in neighborlist])
    #     images = np.array([nbr[3] for nbr in neighborlist])
    #
    #     # 找到 k-th 最近邻距离
    #     max_dist = distances[-1]
    #
    #     # 使用布尔掩码一次性筛选 ids, images 和 distances
    #     mask = distances <= max_dist
    #     ids = ids[mask]
    #     images = images[mask]
    #
    #     # 处理邻居边
    #     for dst, image in zip(ids, images):
    #         src_id, dst_id, src_image, dst_image = canonize_edge(site_idx, dst, (0, 0, 0), tuple(image))
    #         if use_canonize:
    #             edges[(src_id, dst_id)].add(dst_image)
    #         else:
    #             edges[(site_idx, dst)].add(tuple(image))
    #
    #
    # # 如果 use_lattice 为 True，添加自环边，这部分可以放在主循环外部
    # if use_lattice:
    #     lattice_edges = [
    #         tuple(np.array([0, 0, 1])),
    #         tuple(np.array([0, 1, 0])),
    #         tuple(np.array([1, 0, 0])),
    #         tuple(np.array([0, 1, 1])),
    #         tuple(np.array([1, 0, 1])),
    #         tuple(np.array([1, 1, 0]))
    #     ]
    #     for site_idx in range(len(all_neighbors)):  # 遍历所有节点
    #         for edge in lattice_edges:
    #             edges[(site_idx, site_idx)].add(edge)
    # edges_new = edges
    # print("for loop time_new", time.time() - stime, " all_neighbors num:", len(all_neighbors))

    stime = time.time()
    # 使用 LRU 缓存 canonize_edge 的计算结果以减少重复计算
    from functools import lru_cache

    # 假设 canonize_edge 的输入输出比较固定，可以加缓存
    @lru_cache(maxsize=None)
    def canonize_edge_cached(site_idx, dst, zero_tuple, image_tuple):
        return canonize_edge(site_idx, dst, zero_tuple, image_tuple)

    # 继续优化的代码
    edges = defaultdict(set)

    # 提前分配常用变量的空间
    lattice_edges = [
        tuple([0, 0, 1]),
        tuple([0, 1, 0]),
        tuple([1, 0, 0]),
        tuple([0, 1, 1]),
        tuple([1, 0, 1]),
        tuple([1, 1, 0])
    ]

    for site_idx, neighborlist in enumerate(all_neighbors):
        # 找到前 max_neighbors 个最近邻居，使用 heapq.nsmallest
        neighborlist = heapq.nsmallest(max_neighbors, neighborlist, key=lambda x: x[2])

        # 提取距离、id 和图像，尽量避免不必要的 np.array 转换
        distances = [nbr[2] for nbr in neighborlist]
        ids = [nbr[1] for nbr in neighborlist]
        images = [nbr[3] for nbr in neighborlist]

        # 找到 k-th 最近邻距离
        max_dist = distances[-1]

        # 使用布尔掩码一次性筛选 ids, images 和 distances
        mask = [dist <= max_dist for dist in distances]
        ids = [ids[i] for i in range(len(ids)) if mask[i]]
        images = [images[i] for i in range(len(images)) if mask[i]]

        # 处理邻居边
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge_cached(site_idx, dst, (0, 0, 0), tuple(image))
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    # 并行化的自环边添加 (可以考虑使用多线程/多进程)
    if use_lattice:
        for site_idx in range(len(all_neighbors)):  # 遍历所有节点
            for edge in lattice_edges:
                edges[(site_idx, site_idx)].add(edge)
    # edges_new2 = edges
    print("for loop time_new2", time.time() - stime, " all_neighbors num:", len(all_neighbors))


    # stime = time.time()
    # from concurrent.futures import ThreadPoolExecutor
    # # 假设 canonize_edge 的输入输出比较固定，使用 LRU 缓存以减少重复计算
    # @lru_cache(maxsize=None)
    # def canonize_edge_cached(site_idx, dst, zero_tuple, image_tuple):
    #     return canonize_edge(site_idx, dst, zero_tuple, image_tuple)
    #
    # # 初始化存储边的结构
    # edges = defaultdict(set)
    #
    # # 提前分配常用的自环边，避免每次循环时重新生成
    # lattice_edges = [
    #     tuple([0, 0, 1]),
    #     tuple([0, 1, 0]),
    #     tuple([1, 0, 0]),
    #     tuple([0, 1, 1]),
    #     tuple([1, 0, 1]),
    #     tuple([1, 1, 0])
    # ]
    #
    # # 定义处理每个 site_idx 的函数，用于并行化处理
    # def process_neighbors(site_idx, neighborlist):
    #     # 找到前 max_neighbors 个最近邻居，使用 heapq.nsmallest
    #     neighborlist = heapq.nsmallest(max_neighbors, neighborlist, key=lambda x: x[2])
    #
    #     # 提取距离、id 和图像，尽量避免不必要的 np.array 转换
    #     distances = [nbr[2] for nbr in neighborlist]
    #     ids = [nbr[1] for nbr in neighborlist]
    #     images = [nbr[3] for nbr in neighborlist]
    #
    #     # 找到 k-th 最近邻距离
    #     max_dist = distances[-1]
    #
    #     # 使用布尔掩码一次性筛选 ids, images 和 distances
    #     mask = [dist <= max_dist for dist in distances]
    #     ids = [ids[i] for i in range(len(ids)) if mask[i]]
    #     images = [images[i] for i in range(len(images)) if mask[i]]
    #
    #     # 处理邻居边
    #     for dst, image in zip(ids, images):
    #         src_id, dst_id, src_image, dst_image = canonize_edge_cached(site_idx, dst, (0, 0, 0), tuple(image))
    #         if use_canonize:
    #             edges[(src_id, dst_id)].add(dst_image)
    #         else:
    #             edges[(site_idx, dst)].add(tuple(image))
    #
    # # 使用并行化处理每个 neighborlist
    # with ThreadPoolExecutor() as executor:
    #     executor.map(process_neighbors, range(len(all_neighbors)), all_neighbors)
    #
    # # 并行化自环边的添加
    # def add_lattice_edges(site_idx):
    #     for edge in lattice_edges:
    #         edges[(site_idx, site_idx)].add(edge)
    #
    # # 如果 use_lattice 为 True，使用并行化添加自环边
    # if use_lattice:
    #     with ThreadPoolExecutor() as executor:
    #         executor.map(add_lattice_edges, range(len(all_neighbors)))
    #
    # print("for loop time_new3", time.time() - stime, " all_neighbors num:", len(all_neighbors))
    # edges_new3 = edges
    # print("old == new:",edges_old==edges_new,"old == new2:",edges_old==edges_new2,"old == new3:",edges_old==edges_new3)

    return edges

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

    frac_coords = torch.tensor(atoms.frac_coords).requires_grad_().to(torch.float32)
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
    coordinates_cartesian = torch.mm(frac_coords, lattice)
    # coordinates_cartesian = torch.tensor(atoms.cart_coords, dtype=torch.float32).requires_grad_()

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

    return u1, v1, r1, frac_coords
    # return u1, v1, r1, coordinates_cartesian



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
        stime = time.time()
        edges = nearest_neighbor_edges_submit(
            atoms=atoms,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            id=id,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )
        print("nearest_neighbor_edges_submit time", time.time() - stime)
        # u, v, r = build_undirected_edgedata(atoms, edges)
        # u, v, r, cart_coords = build_undirected_edgedata_tensor(atoms, edges)
        u, v, r, coords = build_undirected_edgedata_tensor(atoms, edges)


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
    stime = time.time()
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features=atom_features))  # "cgcnn" or "atomic number"
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    print("build up atom attribute tensor time", time.time() - stime)

    # g = Data(x=node_features, edge_index=edge_index, edge_attr=r)
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=r, node_pos=coords)
    # g = Data(x=node_features, edge_attr=r.norm(dim=-1), edge_index=torch.stack([u, v]), inf_edge_index=inf_edge_index,
    #             inf_edge_attr=inf_edge_attr)

    # if compute_line_graph:
    #     linegraph_trans = LineGraph(force_directed=True)
    #     g_new = Data()
    #     g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
    #     lg = linegraph_trans(g)
    #     lg.edge_attr = pyg_compute_bond_cosines(lg)
    #     return g_new, lg
    # else:
    # load selected node representation
    # assume graphs contain atomic number in g.ndata["atom_features"]
    stime = time.time()
    features = _get_attribute_lookup("cgcnn")

    z = graph.x
    graph.atomic_number = z
    z = z.type(torch.IntTensor).squeeze()
    f = torch.tensor(features[z]).type(torch.FloatTensor)
    if graph.x.size(0) == 1:
        f = f.unsqueeze(0)
    graph.x = f
    print("load selected node representation time", time.time() - stime)
    return graph

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

def convert_AFL_to_graphs(input_lattice,frac_coords,batch):


    # lattice_matrix = inputlattice
    # coords = frac_coords
    elements = atomic_numbers_to_symbols(batch.atom_types.tolist())

    batch_size = input_lattice.shape[0]
    input_lattice = input_lattice.tolist()
    frac_coords = frac_coords.tolist()
    graphs = []
    current_index = 0
    for i in range(batch_size):
        stime = time.time()
        # 获取每个晶体的原子数量
        num_atoms = batch.num_atoms[i].item()

        # 提取对应晶体的晶格矩阵、分数坐标和元素
        lattice_mat = input_lattice[i]  # 3x3 矩阵
        coords = frac_coords[current_index:current_index + num_atoms]  # num_atoms x 3 坐标
        elems = elements[current_index:current_index + num_atoms]  # 对应的元素符号

        # 创建晶体实例
        structure = Atoms(lattice_mat=lattice_mat, coords=coords, elements=elems)

        stime2 = time.time()
        graph = atom_dgl_multigraph(
            structure,
            neighbor_strategy='k-nearest',  #'k-nearest'
            cutoff=8,
            atom_features="atomic_number",
            max_neighbors=12,
            compute_line_graph=False,
            use_canonize=True,
            use_lattice=True,
            use_angle=False,
        )
        print("atom_dgl_multigraph time",time.time()-stime2)
        graphs.append(graph)
        # 更新原子索引
        current_index += num_atoms
        print("convert_time",time.time()-stime)

    batched_graph = Batch.from_data_list(graphs)

    return batched_graph






