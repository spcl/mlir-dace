// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Dangling out-connector
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{
        sdfg.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdfg.return %5 : index
        }
        sdfg.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdfg.return %20 : index
        }
        %n = sdfg.call @five() : () -> index
        %m = sdfg.call @twenty() : () -> index
        %a = sdfg.alloc{transient}(%n, %m) : !sdfg.array<?x?xi32>
    }
}
