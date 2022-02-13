// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<i32>

    sdfg.state @state_0{
        sdfg.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b : i32
            sdfg.return %c : i32
        }
        sdfg.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdfg.return %1 : i32
        }

        %1 = sdfg.call @one() : () -> i32
        %c = sdfg.call @add(%1, %1) : (i32, i32) -> i32

        %a = sdfg.get_access %A : !sdfg.array<i32> -> !sdfg.memlet<i32>
        sdfg.store %c, %a[] : i32 -> !sdfg.memlet<i32>
    }
}
