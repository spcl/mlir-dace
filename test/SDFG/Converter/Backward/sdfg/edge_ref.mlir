// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg (%arg0: index) -> (%r: !sdfg.array<i32>){
  sdfg.state @state_1 {}
  sdfg.state @state_2 {}
  sdfg.state @state_3 {}
  sdfg.state @state_4 {}

  sdfg.edge {assign = ["idx: ref"]} (ref: %arg0: index) @state_1 -> @state_2
  sdfg.edge {condition = "idx < ref"} (ref: %arg0: index) @state_2 -> @state_3
  sdfg.edge {condition = "not(idx < ref)"} (ref: %arg0: index) @state_2 -> @state_4    
}

