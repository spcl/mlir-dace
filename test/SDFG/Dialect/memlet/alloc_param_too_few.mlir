// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        %n = sdfg.tasklet() -> (index) {
            %5 = arith.constant 5 : index
            sdfg.return %5 : index
        }
        %a = sdfg.alloc(%n) : !sdfg.array<?x?xi32>
    }
}
