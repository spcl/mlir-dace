// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<2x6x8xi32>
    %B = sdfg.alloc() : !sdfg.array<2x6x8xi32>
    %C = sdfg.alloc() : !sdfg.array<2x6x8xi32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.array<2x6x8xi32> -> !sdfg.memlet<2x6x8xi32>
        %b = sdfg.get_access %B : !sdfg.array<2x6x8xi32> -> !sdfg.memlet<2x6x8xi32>
        %c = sdfg.get_access %C : !sdfg.array<2x6x8xi32> -> !sdfg.memlet<2x6x8xi32>
        sdfg.tasklet @add(%x: i32, %y: i32) -> i32{
            %z = arith.addi %x, %y : i32
            sdfg.return %z : i32
        }
        sdfg.map (%i, %j, %g) = (0, 0, 0) to (2, 2, 2) step (1, 1, 1) {
            %a_ijg = sdfg.load %a[%i, %j, %g] : !sdfg.memlet<2x6x8xi32> -> i32
            %b_ijg = sdfg.load %b[%i, %j, %g] : !sdfg.memlet<2x6x8xi32> -> i32
            %res = sdfg.call @add(%a_ijg, %b_ijg) : (i32, i32) -> i32
            sdfg.store %res, %c[%i, %j, %g] : i32 -> !sdfg.memlet<2x6x8xi32>
        }
    }
}
