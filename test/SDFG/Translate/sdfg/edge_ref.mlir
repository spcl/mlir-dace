// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdir.sdfg {entry = @state_1} @kernel_2mm(%arg0: index) {
    sdir.state @state_1 {
    }
  
    sdir.state @state_2 {
    }

    sdir.state @state_3 {
    }

    sdir.state @state_4 {
    }

    sdir.edge {assign = ["idx: ref"]} (ref: %arg0: index) @state_1 -> @state_2
    sdir.edge {condition = "idx < ref"} (ref: %arg0: index) @state_2 -> @state_3
    sdir.edge {condition = "not(idx < ref)"} (ref: %arg0: index) @state_2 -> @state_4    
  }
}
