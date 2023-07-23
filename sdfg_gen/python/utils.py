# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich


def export_sdfg(sdfg, name=None):
    if name is None: name = sdfg.name
    else: name = sdfg.name + "_" + name

    out_file_json = "gen/json/" + name + ".json"
    out_file_sdfg = "gen/sdfg/" + name + ".sdfg"

    sdfg.save(filename=out_file_json)
    sdfg.save(filename=out_file_sdfg)
