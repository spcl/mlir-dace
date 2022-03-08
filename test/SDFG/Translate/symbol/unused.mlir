// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{
        sdfg.alloc_symbol("N")
        %res = sdfg.sym("3*N+2") : i64
    }
}
