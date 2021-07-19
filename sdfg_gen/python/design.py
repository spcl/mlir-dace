# This file was used to create the images for the design of SDIR

import dace
from dace import dtypes
from utils import export_sdfg
import numpy as np

# State only
sdfg = dace.SDFG("design")
state = sdfg.add_state()
export_sdfg(sdfg, "state")

# Tasklet
tasklet = state.add_tasklet(name='add',
                            inputs={'a', 'b'},
                            outputs={'c', 'd'},
                            code='c = a + b\nd = a - b',
                            language=dace.Language.Python)

export_sdfg(sdfg, "tasklet")

# Parallel Memlets
state.remove_node(tasklet)

A = state.add_read('A')
B = state.add_read('B')

C = state.add_write('C')
D = state.add_write('D')

e1 = state.add_edge(A, None, C, None, dace.Memlet('A[0]'))
e2 = state.add_edge(B, None, D, None, dace.Memlet('B[0]'))

export_sdfg(sdfg, "parallel")

# Memlets
state.remove_edge(e1)
state.remove_edge(e2)
state.remove_node(D)
export_sdfg(sdfg, "memlets")

# Full Graph
tasklet = state.add_tasklet(name='add',
                            inputs={'a', 'b'},
                            outputs={'c'},
                            code='c = a + b',
                            language=dace.Language.Python)

state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
state.add_edge(B, None, tasklet, 'b', dace.Memlet('B[0]'))
state.add_edge(tasklet, 'c', C, None, dace.Memlet('C[0]'))

export_sdfg(sdfg, "full")

# Self-write
sdfg = dace.SDFG("design")
state = sdfg.add_state()

A = state.add_read('A')
A2 = state.add_write('A')

tasklet = state.add_tasklet(name='add',
                            inputs={'a'},
                            outputs={'a_1'},
                            code='a_1 = a + 1',
                            language=dace.Language.Python)

state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
state.add_edge(tasklet, 'a_1', A2, None, dace.Memlet('A[0]'))
export_sdfg(sdfg, "self-write")

# data race
sdfg = dace.SDFG("design")
state = sdfg.add_state()

A = state.add_read('A')
A2 = state.add_write('A')
B = state.add_read('B')
C = state.add_write('C')

tasklet = state.add_tasklet(name='add',
                            inputs={'a'},
                            outputs={'c'},
                            code='c = a + 1',
                            language=dace.Language.Python)

tasklet2 = state.add_tasklet(name='add',
                            inputs={'b'},
                            outputs={'a'},
                            code='a = b + 1',
                            language=dace.Language.Python)

state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
state.add_edge(tasklet, 'c', C, None, dace.Memlet('C[0]'))
state.add_edge(B, None, tasklet2, 'b', dace.Memlet('B[0]'))
state.add_edge(tasklet2, 'a', A2, None, dace.Memlet('A[0]'))
export_sdfg(sdfg, "data_race")

# Map
sdfg = dace.SDFG("design")
state = sdfg.add_state()
A = state.add_read('A')
B = state.add_read('B')
C = state.add_write('C')

tasklet, map_entry, map_exit = state.add_mapped_tasklet(
    name='add',                                         
    map_ranges=dict(i='0:2', j='0:2'),                             
    inputs=dict(a=dace.Memlet('A[i, j]'), b=dace.Memlet('B[i, j]')), 
    code='c = a + b',                                   
    outputs=dict(c=dace.Memlet('C[i, j]'))
)

state.add_edge(A, None, map_entry, None, memlet=dace.Memlet('A[0:2,0:2]'))
state.add_edge(B, None, map_entry, None, memlet=dace.Memlet('B[0:2,0:2]'))
state.add_edge(map_exit, None, C, None, memlet=dace.Memlet('C[0:2,0:2]'))

sdfg.fill_scope_connectors()
export_sdfg(sdfg, "map")

# nested & lib
@dace.program
def mmm(A, B):
    return A @ B

@dace.program
def design_nested(A, B):
    C = mmm(A, B)

a = np.random.rand(2, 2)
sdfg = design_nested.to_sdfg(a, a)
export_sdfg(sdfg)

# Symbols
sdfg = dace.SDFG("design")
state = sdfg.add_state()
A = state.add_read('A')
C = state.add_write('C')

tasklet, map_entry, map_exit = state.add_mapped_tasklet(
    name='add',                                         
    map_ranges=dict(i='0:N'),                             
    inputs=dict(a=dace.Memlet('A[i]')), 
    code='c = a + 1',                                   
    outputs=dict(c=dace.Memlet('C[i]'))
)

state.add_edge(A, None, map_entry, None, memlet=dace.Memlet('A[0:N]'))
state.add_edge(map_exit, None, C, None, memlet=dace.Memlet('C[0:N]'))

sdfg.fill_scope_connectors()
export_sdfg(sdfg, "sym")

# Multistate
sdfg = dace.SDFG("design")
state = sdfg.add_state(is_start_state=True)
state2 = sdfg.add_state_after(state)
state3 = sdfg.add_state_after(state2)
state4 = sdfg.add_state()


sdfg.add_edge(state2, state4, dace.InterstateEdge())

export_sdfg(sdfg, "multistate")

# Streams
sdfg = dace.SDFG("design")
state = sdfg.add_state()
A = state.add_stream('A', dtypes.int32)
#A_2 = state.add_stream('A', dtypes.int32)
C = state.add_write('C')

consEntry, consExit = state.add_consume("add_one", ("p","P"), "A = 0")

tasklet = state.add_tasklet(name='add',
                            inputs={'a'},
                            outputs={'c'},
                            code='c = a + 1',
                            language=dace.Language.Python)

state.add_edge(A, None, consEntry, None, memlet=dace.Memlet('A[0:2]'))
state.add_edge(consEntry, None, tasklet, None, memlet=dace.Memlet('A[p]'))
state.add_edge(tasklet, None, consExit, None, memlet=dace.Memlet('C[p]'))
state.add_edge(consExit, None, C, None, memlet=dace.Memlet('C[0:2]'))
#state.add_edge(consExit, None, A_2, None, memlet=dace.Memlet('A[0:2]'))

export_sdfg(sdfg, "stream")

@dace.program
def design_ex(in1: dtypes.vector(dtypes.int32,5)):
    sum = 0

    for i in range(5):
        sum = sum + in1[i]

    return sum

sdfg = design_ex.to_sdfg()
export_sdfg(sdfg)

'''
@dace.program
def design_fail(in1: dtypes.vector(dtypes.int32,5)):
    return in1[0] + in1[1]

sdfg = design_fail.to_sdfg()
export_sdfg(sdfg)
'''