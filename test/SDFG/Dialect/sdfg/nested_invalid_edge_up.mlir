// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: does not reference a valid state

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_1{
  }

  sdfg.state @state_0{
    %N = sdfg.alloc() : !sdfg.array<i32>

    sdfg.nested_sdfg{entry=@state_1} () -> (%N: !sdfg.array<i32>){
      sdfg.state @state_2{
      }

      sdfg.edge @state_2 -> @state_1
    }
  }
} 
