// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Dangling out-connector

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        %n = sdfg.tasklet() -> index{
                %5 = arith.constant 5 : index
                sdfg.return %5 : index
            }

        %m = sdfg.tasklet() -> index{
                %20 = arith.constant 20 : index
                sdfg.return %20 : index
            }

        %a = sdfg.alloc(%n, %m) : !sdfg.array<?x?xi32>
    }
}
