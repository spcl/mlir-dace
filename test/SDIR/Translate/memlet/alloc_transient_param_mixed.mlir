// RUN: sdir-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Dangling out-connector
sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdir.return %5 : index
        }
        sdir.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdir.return %20 : index
        }
        %n = sdir.call @five() : () -> index
        %m = sdir.call @twenty() : () -> index
        %a = sdir.alloc_transient(%n, %m) : !sdir.array<?x6x?xi32>
    }
}
