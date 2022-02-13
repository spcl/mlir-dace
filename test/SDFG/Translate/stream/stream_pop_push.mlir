// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream_array<i32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.stream_array<i32> -> !sdfg.stream<i32>
        
        sdfg.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdfg.return %1 : i32
        }

        %1 = sdfg.call @one() : () -> i32

        sdfg.stream_push %1, %a : i32 -> !sdfg.stream<i32>
        %a_1 = sdfg.stream_pop %a : !sdfg.stream<i32> -> i32
    }
}
