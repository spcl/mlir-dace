// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc_stream() : !sdir.stream_array<i32>
    %C = sdir.alloc() : !sdir.array<6xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>
        %c = sdir.get_access %C : !sdir.array<6xi32> -> !sdir.memlet<6xi32>
        sdir.tasklet @add_one(%x: i32) -> i32{
            %1 = arith.constant 1 : i32
            %res = arith.addi %x, %1 : i32
            sdir.return %res : i32
        }
        builtin.func @empty(%x: !sdir.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdir.stream_length %x : !sdir.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }
        sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        sdir.consume{num_pes=5, condition=@empty} (%a : !sdir.stream<i32>) -> (pe: %p, elem: %e) {
            %res = sdir.call @add_one(%e) : (i32) -> i32
            %0 = sdir.call @zero() : () -> index
            sdir.store{wcr="add"} %res, %c[%0] : i32 -> !sdir.memlet<6xi32>
        }
    }
}
