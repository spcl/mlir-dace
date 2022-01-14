func private @const42(%out: memref<i32>){
  %c42 = arith.constant 42 : i32
  memref.store %c42, %out[] : memref<i32>
  return
}
