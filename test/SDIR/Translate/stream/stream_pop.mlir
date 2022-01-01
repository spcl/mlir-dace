// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc_stream() : !sdir.stream_array<i32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>
        %a_1 = sdir.stream_pop %a : !sdir.stream<i32> -> i32
    }
}
