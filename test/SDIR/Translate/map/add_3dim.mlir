// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x6x8xi32>
    %B = sdir.alloc() : !sdir.array<2x6x8xi32>
    %C = sdir.alloc() : !sdir.array<2x6x8xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<2x6x8xi32> -> !sdir.memlet<2x6x8xi32>
        %b = sdir.get_access %B : !sdir.array<2x6x8xi32> -> !sdir.memlet<2x6x8xi32>
        %c = sdir.get_access %C : !sdir.array<2x6x8xi32> -> !sdir.memlet<2x6x8xi32>
        sdir.tasklet @add(%x: i32, %y: i32) -> i32{
            %z = arith.addi %x, %y : i32
            sdir.return %z : i32
        }
        sdir.map (%i, %j, %g) = (0, 0, 0) to (2, 2, 2) step (1, 1, 1) {
            %a_ijg = sdir.load %a[%i, %j, %g] : !sdir.memlet<2x6x8xi32> -> i32
            %b_ijg = sdir.load %b[%i, %j, %g] : !sdir.memlet<2x6x8xi32> -> i32
            %res = sdir.call @add(%a_ijg, %b_ijg) : (i32, i32) -> i32
            sdir.store %res, %c[%i, %j, %g] : i32 -> !sdir.memlet<2x6x8xi32>
        }
    }
}
