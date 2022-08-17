// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.nested_sdfg () -> (%r: !sdfg.array<i32>) {
      %3 = sdfg.alloc() : !sdfg.array<i32>

      sdfg.state @state_1{
        %0 = sdfg.load %r[] : !sdfg.array<i32> -> i32
        sdfg.store %0, %3[] : i32 -> !sdfg.array<i32>
      }

      sdfg.state @state_2{
        %0 = sdfg.load %3[] : !sdfg.array<i32> -> i32
        sdfg.store %0, %r[] : i32 -> !sdfg.array<i32>
      }

      sdfg.edge {assign = [], condition = "1"} @state_1 -> @state_2
    }
  }
} 
