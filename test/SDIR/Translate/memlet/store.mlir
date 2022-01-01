// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<i32>
    sdir.state @state_0 {
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        %1 = sdir.call @one() : () -> i32
        %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
        sdir.store %1, %a[] : i32 -> !sdir.memlet<i32>
    }
}
