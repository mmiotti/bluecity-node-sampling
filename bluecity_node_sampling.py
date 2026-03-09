import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import osmnx as ox
import time
import logging

from scipy.stats import lognorm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def timeit(fn):
    """Time function and log execution time (use as a decorator)."""
    def timed(*args, **kw):
        t1 = time.perf_counter()
        result = fn(*args, **kw)
        t2 = time.perf_counter()
        logging.info(f'Function `{fn.__name__}` completed; it took {t2 - t1:.2f} seconds.')
        return result
    return timed


def show_weight_info(lognorm_mu: float,
                     lognorm_sigma: float):
    """Show some info on how time-based weights for a given node pair are calculated."""
    max_time = np.exp(lognorm_mu - lognorm_sigma**2)
    logging.info(f"Maximum weight is assigned to two nodes that are {max_time:.0f} s ({max_time/60:.1f} minutes) apart")
    times = [1, 2, 5, 10, 30, 60]
    weights = lognorm.pdf([max_time] + [t * 60 for t in times], s=lognorm_sigma, scale=np.exp(lognorm_mu))
    weights = weights / weights[0]
    logging.info(f"Relative weights for {', '.join([f'{t} min: {w:.2f}' for t, w in zip(times, weights[1:])])}")


@timeit
def load_network() -> nx.MultiDiGraph:
    file_name = 'lausanne_drive.graphml'
    g = ox.load_graphml(file_name)
    return g


@timeit
def load_edge_attributes(g: nx.MultiDiGraph) -> tuple[pd.Series, pd.Series]:
    """Obtain length and speed edge attributes from networkx graph."""

    length = pd.Series(nx.get_edge_attributes(g, 'length'), name='length')
    speed_kph = pd.Series(nx.get_edge_attributes(g, 'speed_kph'), name='speed_kph')
    return length, speed_kph


@timeit
def assign_edge_weight(g: nx.MultiDiGraph,
                       weight_name: str,
                       length: pd.Series,
                       speed_kph: pd.Series,
                       betweenness: None | pd.Series,
                       betweenness_to_slowdown: None | int | float):
    """Assign edge weight to networkx graph representing travel time on each edge.

    If `betweenness` and `betweenness_to_slowdown` are provided, betweenness centrality is taken into account.
    A betweenness value of `betweenness_to_slowdown` leads to a speed decrease of 50%. For example, if
    betweenness_to_slowdown = 10_000 and betweenness = 5_000, the speed on that edge is decreased by 1 / (1 + 0.5) = 33%.
    """

    if betweenness is not None and betweenness_to_slowdown:
        speed_kph = speed_kph / (1 + betweenness / betweenness_to_slowdown)
    weight = length / (speed_kph / 3.6)
    nx.set_edge_attributes(g, weight.to_dict(), weight_name)
    return g


@timeit
def get_considered_nodes(g: nx.MultiDiGraph,
                         rng: np.random.RandomState,
                         max_nodes: int,
                         node_weight_col: str) -> pd.Series:
    """Get nodes to be used in travel time matrix estimation with each value representing each node's static weight.

    These nodes are then available for final od pair sampling.
    """

    n = ox.graph_to_gdfs(g, nodes=True, edges=False)
    if node_weight_col == 'dummy':
        n[node_weight_col] = 1
    n = n.loc[(n['street_count'] >= 3) & (n[node_weight_col] > 0), node_weight_col]
    logging.info(f'{len(n):,} nodes are available for sampling.')
    if len(n) > max_nodes:
        n = n.sample(max_nodes, random_state=rng, replace=False, weights=n)
    return n


@timeit
def networkx_to_igraph_with_indices(g: nx.MultiDiGraph) -> tuple[ig.Graph, dict[str, dict]]:
    """Convert networkx graph to igraph graph along with dictionary-based index maps."""

    e = ox.graph_to_gdfs(g, nodes=False, edges=True)
    nx.set_edge_attributes(g, {idx: idx for idx in e.index}, name='nx_edge_id')
    h = ig.Graph.from_networkx(g)
    idx_maps = {'node_nx_to_ig': {a: b for a, b in zip(h.vs()['_nx_name'], h.vs.indices)},
                'node_ig_to_nx': {b: a for a, b in zip(h.vs()['_nx_name'], h.vs.indices)},
                'edge_nx_to_ig': {a: b for a, b in zip(h.es()['nx_edge_id'], h.get_edgelist())},
                'edge_ig_to_nx': {b: a for a, b in zip(h.es()['nx_edge_id'], h.get_edgelist())}}
    return h, idx_maps


@timeit
def edge_betweenness_igraph(h: ig.Graph,
                            expected_km_driven: int | float,
                            directed: bool = True,
                            cutoff: None | int | float = None,
                            weights: None | str = None,
                            sources: None | list[int] = None,
                            targets: None | list[int] = None) -> dict[int, float]:
    """Calculate normalized edge betweenness centralities using iGraph."""

    bc_result = h.edge_betweenness(directed, cutoff, weights, sources, targets)
    # Normalize betweenness
    total_sum = sum([bc * length for bc, length in zip(bc_result, h.es['length'])])
    factor = expected_km_driven * 1_000 / total_sum
    bc_dict = {idx: bc * factor for idx, bc in zip(h.get_edgelist(), bc_result)}
    return bc_dict


@timeit
def travel_time_matrix_igraph(h, nodes, weight_name) -> list[list[int]]:
    return h.distances(source=nodes, target=nodes, weights=weight_name)


# @timeit
# def travel_time_matrix_nx(g, weight_name, t_matrix_dict):
#     """ONLY FOR TESTING"""
#     t_nx = {}
#     for orig, dests in t_matrix_dict.items():
#         t_nx[orig] = {}
#         for dest in dests.keys():
#             t_nx[orig][dest] = nx.shortest_path_length(g, orig, dest, weight=weight_name)
#     return t_nx


@timeit
def igraph_matrix_to_dict(t_matrix: list[list[float]],
                          nodes_ig: list[int],
                          idx_maps) -> dict[int, dict[int, float]]:
    """Convert igraph travel time matrix result (list of lists) to dict of dicts, with node ids as keys."""

    t_matrix_dict = {}
    for row_id, t_list in zip(nodes_ig, t_matrix):
        d = {idx_maps['node_ig_to_nx'][col_id]: t for col_id, t in zip(nodes_ig, t_list)}
        t_matrix_dict[idx_maps['node_ig_to_nx'][row_id]] = d
    return t_matrix_dict


@timeit
def sample_od_pairs(nodes: pd.Series,
                    rng: np.random.RandomState,
                    n_samples: int,
                    node_weight_col: str,
                    lognorm_mu: float,
                    lognorm_sigma: float,
                    t_matrix_dict: dict[int, dict[int, float]]) -> dict[int, list[int]]:
    """Sample set of OD pairs to be used to calculate final flows on network."""

    origins = list(nodes.sample(n_samples, random_state=rng, replace=False, weights=nodes).index)
    od_pairs = {}
    for origin in origins:
        times = [t_matrix_dict[origin][dest] for dest in nodes.index]
        time_weights = lognorm.pdf(times, s=lognorm_sigma, scale=np.exp(lognorm_mu))
        weights = nodes * time_weights
        try:
            od_pairs[origin] = list(nodes.sample(n_samples, random_state=rng, replace=False, weights=weights).index)
        except ValueError:
            logging.info(f'Failed to sample destinations for origin {origin}. Likely, travel times could not be '
                         f'computed (is node not connected to network?)')
    return od_pairs


@timeit
def main():
    # Maximum number of nodes for which travel time matrix is calculated and which are considered in sampling process.
    n_nodes_preprocess = 1000

    # Number of origins and destinations to be sampled (among available nodes, see above) for final route calculations.
    n_nodes_od = 500

    # Factor to normalize betweenness centrality regardless of how many nodes are sampled to vehicles/edge/day. This
    # number should represent the expected total number of vehicle-km driven on the network per day. It should be a
    # function of population, estimated car ownership (cars/person), and estimated km driven on the network per day
    # by each car. E.g. 250,000 (population) * 0.5 (ownership) * 10 (km/car/day) = 1250000
    daily_km_driven = 1_250_000

    # At the following betweenness value (veh/day), travel times on that edge are doubled (50% slowdown).
    betweenness_to_slowdown = 20_000

    # Column that determines static node weights (e.g. population and employment near each node). If set to 'dummy',
    # all node weights are 1.
    node_weight_col = 'dummy'

    # Lognormal PDF coefficients that determine sample weight adjustment for each node based on travel time to origin
    # node. These coefficients have been obtained by fitting data from the Swiss Mobility and Transport Microcensus
    # (car trips only; to/from/within Lausanne).
    lognorm_mu = 6.85
    lognorm_sigma = 0.83

    # Edge weight names to be used for first iteration (to calculate betweenness centrality) and final weights.
    edge_weight_default = 'duration'
    edge_weight_bc = 'duration_bc'

    # Seed number for random state (to make results consistent).
    seed = 42

    # OPTIONAL: show some information on time-based weighting function.
    show_weight_info(lognorm_mu, lognorm_sigma)

    if n_nodes_preprocess < n_nodes_od * 1.05:
        raise ValueError("Number of nodes for which travel time matrix is calculated should be at least 5% higher than "
                         "desired number of origins/destinations in od pairs.")

    # Prepare data
    rng = np.random.RandomState(seed)
    g = load_network()
    length, speed_kph = load_edge_attributes(g)
    g = assign_edge_weight(g, edge_weight_default, length, speed_kph, None, None)
    nodes = get_considered_nodes(g, rng, n_nodes_preprocess, node_weight_col)
    h, idx_maps = networkx_to_igraph_with_indices(g)
    nodes_ig = [idx_maps['node_nx_to_ig'][idx] for idx in nodes.index]

    # Calculate betweenness centrality using matrix node sample
    bc_dict = edge_betweenness_igraph(h, daily_km_driven, weights=edge_weight_default,
                                      sources=nodes_ig, targets=nodes_ig)
    betweenness = {idx_maps['edge_ig_to_nx'][idx]: bc for idx, bc in bc_dict.items()}
    betweenness = pd.Series({k: betweenness.get(k, 0) for k in length.index}, name='betweenness')
    # Re-calculate edge weights using betweenness centrality and assign those values to igraph network as well
    g = assign_edge_weight(g, edge_weight_bc, length, speed_kph, betweenness, betweenness_to_slowdown)
    duration = nx.get_edge_attributes(g, edge_weight_bc)
    h.es[edge_weight_bc] = [duration[idx_maps['edge_ig_to_nx'][idx]] for idx in h.get_edgelist()]

    # Calculate travel time matrix and express matrix in terms of networkx IDs
    t_matrix = travel_time_matrix_igraph(h, nodes_ig, edge_weight_bc)
    t_matrix_dict = igraph_matrix_to_dict(t_matrix, nodes_ig, idx_maps)

    # ONLY FOR TESTING: Verify that igraph distance matrix produced (including node mapping back to nx) matches expected results
    # t_matrix_nx = travel_time_matrix_nx(g, edge_weight_bc, t_matrix_dict)

    # Sample final OD pairs based on travel time matrix
    od_pairs = sample_od_pairs(nodes, rng, n_nodes_od, node_weight_col, lognorm_mu, lognorm_sigma, t_matrix_dict)
    # do something with od_pairs...


if __name__ == '__main__':
    main()