// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %a = sdfg.alloc() : !sdfg.array<23x45x123xi32>

  sdfg.state @state_0{
  }
} 
