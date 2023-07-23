# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

# Taken and modified from: https://github.com/spcl/dace/blob/master/tests/nest_subgraph_test.py

import dace
from dace.sdfg.nodes import MapEntry, Tasklet
from dace.sdfg.graph import NodeNotFoundError, SubgraphView
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.dataflow import tiling


def create_sdfg():
    sdfg = dace.SDFG('badscope_test')
    sdfg.add_array('A', [2], dace.float32)
    sdfg.add_array('B', [2], dace.float32)
    state = sdfg.add_state()
    t, me, mx = state.add_mapped_tasklet('map',
                                         dict(i='0:2'),
                                         dict(a=dace.Memlet.simple('A', 'i')),
                                         'b = a * 2',
                                         dict(b=dace.Memlet.simple('B', 'i')),
                                         external_edges=True)
    return sdfg, state, t, me, mx


sdfg, state, t, me, mx = create_sdfg()
nest_state_subgraph(sdfg, state, SubgraphView(state, [t]))

from utils import export_sdfg

export_sdfg(sdfg)
