// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream()
    // CHECK-SAME: !sdir.stream_array<i32>
    %A = sdir.alloc_stream() : !sdir.stream_array<i32>
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<6xi32>
    %C = sdir.alloc() : !sdir.array<6xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.stream_array<i32> -> !sdir.stream<i32>
        %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>
        // CHECK-NEXT: [[NAMEc:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEC]] 
        // CHECK-SAME: !sdir.array<6xi32> -> !sdir.memlet<6xi32>
        %c = sdir.get_access %C : !sdir.array<6xi32> -> !sdir.memlet<6xi32>
        // CHECK: sdir.tasklet @add_one
        sdir.tasklet @add_one(%x: i32) -> i32{
            %1 = constant 1 : i32
            %res = addi %x, %1 : i32
            sdir.return %res : i32
        }
        // CHECK: func @empty
        func @empty(%x: !sdir.stream<i32>) -> i1{
            %0 = constant 0 : i32
            %length = sdir.stream_length %x : !sdir.stream<i32> -> i32
            %isZero = cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }
        %0 = constant 0 : index
        // CHECK: sdir.consume
        // CHECK-DAG: num_pes = 5
        // CHECK-DAG: condition = @empty
        // CHECK-SAME: [[NAMEa]] : !sdir.stream<i32>
        // CHECK-SAME: pesid: [[NAMEp:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: elem: [[NAMEe:%[a-zA-Z0-9_]*]]
        sdir.consume{num_pes=5, condition=@empty} (%a : !sdir.stream<i32>) -> (pesid: %p, elem: %e) {
            // CHECK-NEXT: [[NAMEres:%[a-zA-Z0-9_]*]] = sdir.call @add_one([[NAMEe]])
            %res = sdir.call @add_one(%e) : (i32) -> i32
            // CHECK-NEXT: sdir.store {wcr = "add"} [[NAMEres]], [[NAMEc]]
            sdir.store{wcr="add"} %res, %c[%0] : i32 -> !sdir.memlet<6xi32>
        }
    }
}
