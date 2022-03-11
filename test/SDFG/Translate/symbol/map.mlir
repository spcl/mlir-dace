// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Dangling out-connector

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0 {
        sdfg.alloc_symbol("N")

        %0 = sdfg.tasklet() -> (index) {
                %0 = arith.constant 0 : index
                sdfg.return %0 : index
            }

        %1 = sdfg.tasklet() -> (index) {
                %1 = arith.constant 1 : index
                sdfg.return %1 : index
            }

        sdfg.map (%i, %j) = (%1, 0) to (2, sym("N")) step (%0, sym("N+2")) {
        }
    }
}
