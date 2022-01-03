// RUN: sdir-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py %s 2>&1 | FileCheck %s
// CHECK: Dangling out-connector
sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x6xi32>
    sdir.state @state_0 {
        sdir.alloc_symbol("N")
        sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        sdir.tasklet @one() -> index{
            %1 = arith.constant 1 : index
            sdir.return %1 : index
        }
        %0 = sdir.call @zero() : () -> index
        %1 = sdir.call @one() : () -> index
        %a = sdir.get_access %A : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        sdir.map (%i, %j) = (%1, 0) to (2, sym("N")) step (%0, sym("N+2")) {
        }
    }
}
