// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK-LABEL: sdir.state @state_0
sdir.state @state_0{

}

// CHECK-LABEL: sdir.state @randomName
sdir.state @randomName{

}