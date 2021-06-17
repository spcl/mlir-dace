def export_sdfg(sdfg):
    out_file_json = "gen/json/" + sdfg.name + ".json"
    out_file_sdfg = "gen/sdfg/" + sdfg.name + ".sdfg"

    sdfg.save(filename=out_file_json, use_pickle=False, with_metadata=False, hash=None, exception=None)
    sdfg.save(filename=out_file_sdfg, use_pickle=False, with_metadata=False, hash=None, exception=None)