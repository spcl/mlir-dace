// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.state @randomName
sdir.state @randomName{

}