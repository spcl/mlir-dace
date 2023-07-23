# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

from dace import SDFG

sdfg = SDFG("single_empty_state")
state = sdfg.add_state()

from utils import export_sdfg

export_sdfg(sdfg)
