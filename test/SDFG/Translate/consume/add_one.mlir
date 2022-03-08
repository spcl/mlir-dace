// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream_array<i32>
    %C = sdfg.alloc() : !sdfg.array<6xi32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.stream_array<i32> -> !sdfg.stream<i32>
        %c = sdfg.get_access %C : !sdfg.array<6xi32> -> !sdfg.memlet<6xi32>
        sdfg.tasklet @add_one(%x: i32) -> i32{
            %1 = arith.constant 1 : i32
            %res = arith.addi %x, %1 : i32
            sdfg.return %res : i32
        }
        builtin.func @empty(%x: !sdfg.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdfg.stream_length %x : !sdfg.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }
        sdfg.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdfg.return %0 : index
        }
        sdfg.consume{num_pes=5, condition=@empty} (%a : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
            %res = sdfg.call @add_one(%e) : (i32) -> i32
            %0 = sdfg.call @zero() : () -> index
            sdfg.store{wcr="add"} %res, %c[%0] : i32 -> !sdfg.memlet<6xi32>
        }
    }
}
