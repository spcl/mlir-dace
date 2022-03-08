// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
module  {
  sdfg.sdfg {entry = @state_1} @kernel_2mm(%arg0: index) {
    sdfg.state @state_1 {
    }
  
    sdfg.state @state_2 {
    }

    sdfg.state @state_3 {
    }

    sdfg.state @state_4 {
    }

    sdfg.edge {assign = ["idx: ref"]} (ref: %arg0: index) @state_1 -> @state_2
    sdfg.edge {condition = "idx < ref"} (ref: %arg0: index) @state_2 -> @state_3
    sdfg.edge {condition = "not(idx < ref)"} (ref: %arg0: index) @state_2 -> @state_4    
  }
}
