# This file generates the needed illustration for the thesis
import dace
from dace import dtypes
from utils import export_sdfg
import numpy as np

# State only
sdfg = dace.SDFG("dr")
state = sdfg.add_state()


# Parallel Memlets

A1= state.add_read('A')
B = state.add_read('B')

C = state.add_write('C')
A2 = state.add_write('A')

e1 = state.add_edge(B, None, A2, None, dace.Memlet('B[0]'))
e2 = state.add_edge(A1, None, C, None, dace.Memlet('A1[0]'))

export_sdfg(sdfg, "data-race")
