def export_sdfg(sdfg, name=None):
    if name is None: name = sdfg.name
    else: name = sdfg.name + "_" + name

    out_file_json = "gen/json/" + name + ".json"
    out_file_sdfg = "gen/sdfg/" + name + ".sdfg"

    sdfg.save(filename=out_file_json, use_pickle=False, with_metadata=False, hash=None, exception=None)
    sdfg.save(filename=out_file_sdfg, use_pickle=False, with_metadata=False, hash=None, exception=None)