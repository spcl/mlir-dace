// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: does not reference a valid func

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.stream<2x6xi32>
  sdfg.state @state_0 {
    func.func @empty(%x: !sdfg.stream<2x6xi32>) -> i1{
      %0 = arith.constant 0 : i32
      %l = sdfg.stream_length %x : !sdfg.stream<2x6xi32> -> i32
      %isZero = arith.cmpi "eq", %l, %0 : i32
      return %isZero : i1
    }

    sdfg.consume{num_pes=5, condition=@full} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
    }
  }
}
