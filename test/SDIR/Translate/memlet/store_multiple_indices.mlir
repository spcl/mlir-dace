// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<56x45xi32>
    sdir.state @state_0 {
        sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        %0 = sdir.call @zero() : () -> index
        %1 = sdir.call @one() : () -> i32
        %a = sdir.get_access %A : !sdir.array<56x45xi32> -> !sdir.memlet<56x45xi32>
        sdir.store %1, %a[%0, %0] : i32 -> !sdir.memlet<56x45xi32>
    }
} 
