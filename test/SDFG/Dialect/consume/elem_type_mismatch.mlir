// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: different type than prior uses: 'i64' vs 'i32'

sdfg.sdfg{entry=@state_0} {
    %A = sdfg.alloc() : !sdfg.stream<i32>
    %C = sdfg.alloc() : !sdfg.array<6xi64>

    sdfg.state @state_0 {
        builtin.func @empty(%x: !sdfg.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdfg.stream_length %x : !sdfg.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }

        sdfg.consume{num_pes=5, condition=@empty} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
            %res = sdfg.tasklet @add_one(%e: i64) -> i64{
                %1 = arith.constant 1 : i64
                %res = arith.addi %e, %1 : i64
                sdfg.return %res : i64
            }

            %0 = sdfg.tasklet @zero() -> index{
                %0 = arith.constant 0 : index
                sdfg.return %0 : index
            }

            sdfg.store{wcr="add"} %res, %C[%0] : i64 -> !sdfg.array<6xi64>
        }
    }
}
