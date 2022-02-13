// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: different type than prior uses: 'i64' vs 'i32'

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.stream<i32>
    %C = sdir.alloc() : !sdir.array<6xi64>

    sdir.state @state_0 {
        builtin.func @empty(%x: !sdir.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdir.stream_length %x : !sdir.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }

        sdir.consume{num_pes=5, condition=@empty} (%A : !sdir.stream<i32>) -> (pe: %p, elem: %e) {
            %res = sdir.tasklet @add_one(%e: i64) -> i64{
                %1 = arith.constant 1 : i64
                %res = arith.addi %e, %1 : i64
                sdir.return %res : i64
            }

            %0 = sdir.tasklet @zero() -> index{
                %0 = arith.constant 0 : index
                sdir.return %0 : index
            }

            sdir.store{wcr="add"} %res, %C[%0] : i64 -> !sdir.array<6xi64>
        }
    }
}
