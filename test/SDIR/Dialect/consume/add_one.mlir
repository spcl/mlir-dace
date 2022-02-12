// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.stream<i32>
    %A = sdir.alloc() : !sdir.stream<i32>
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<6xi32>
    %C = sdir.alloc() : !sdir.array<6xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: builtin.func @empty
        builtin.func @empty(%x: !sdir.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdir.stream_length %x : !sdir.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }
        // CHECK: sdir.consume
        // CHECK-DAG: num_pes = 5
        // CHECK-DAG: condition = @empty
        // CHECK-SAME: [[NAMEA]] : !sdir.stream<i32>
        // CHECK-SAME: pe: [[NAMEp:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: elem: [[NAMEe:%[a-zA-Z0-9_]*]]
        sdir.consume{num_pes=5, condition=@empty} (%A : !sdir.stream<i32>) -> (pe: %p, elem: %e) {
            // CHECK: [[NAMEres:%[a-zA-Z0-9_]*]] = sdir.tasklet @add_one
            // CHECK-SAME: [[NAMEe]]
            %res = sdir.tasklet @add_one(%e: i32) -> i32{
                %1 = arith.constant 1 : i32
                %res = arith.addi %e, %1 : i32
                sdir.return %res : i32
            }
            // CHECK: [[NAMEzero:%[a-zA-Z0-9_]*]] = sdir.tasklet @zero()
            %0 = sdir.tasklet @zero() -> index{
                %0 = arith.constant 0 : index
                sdir.return %0 : index
            }
            // CHECK: sdir.store {wcr = "add"} [[NAMEres]], [[NAMEC]]
            // CHECK-SAME: [[NAMEzero]]
            sdir.store{wcr="add"} %res, %C[%0] : i32 -> !sdir.array<6xi32>
        }
    }
}
