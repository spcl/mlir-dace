// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc_stream() : !sdir.stream_array<i32>
    sdir.state @state_0 {
        sdir.tasklet @zero() -> i32{
            %0 = arith.constant 0 : i32
            sdir.return %0 : i32
        }
        %0 = sdir.call @zero() : () -> i32
        %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>
        sdir.stream_push %0, %a : i32 -> !sdir.stream<i32>
    }
}
