// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<8x16x4xi32>

  sdfg.state @state_0 {
    %a_s = sdfg.subview %A[3, 4, 2][1, 6, 3][1, 1, 1] : !sdfg.array<8x16x4xi32> -> !sdfg.array<6x3xi32>
  }
}
