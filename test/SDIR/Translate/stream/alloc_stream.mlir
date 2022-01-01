// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdir.alloc_stream() : !sdir.stream_array<i32>

    sdir.state @state_0{
    }
}
