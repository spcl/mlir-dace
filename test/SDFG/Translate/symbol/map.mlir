// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Dangling out-connector
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>
    sdfg.state @state_0 {
        sdfg.alloc_symbol("N")
        sdfg.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdfg.return %0 : index
        }
        sdfg.tasklet @one() -> index{
            %1 = arith.constant 1 : index
            sdfg.return %1 : index
        }
        %0 = sdfg.call @zero() : () -> index
        %1 = sdfg.call @one() : () -> index
        %a = sdfg.get_access %A : !sdfg.array<2x6xi32> -> !sdfg.memlet<2x6xi32>
        sdfg.map (%i, %j) = (%1, 0) to (2, sym("N")) step (%0, sym("N+2")) {
        }
    }
}
