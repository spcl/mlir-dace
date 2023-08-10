// XFAIL: *
// RUN: mlir-opt --linalg-bufferize --func-bufferize --cse --finalizing-bufferize %s | sdfg-opt --linalg-to-sdfg | sdfg-opt
func.func @main() -> tensor<256x256xf32> {
  %0 = linalg.init_tensor [256, 16] : tensor<256x16xf32>
  %1 = linalg.init_tensor [16, 256] : tensor<16x256xf32>
  %2 = linalg.init_tensor [256, 256] : tensor<256x256xf32>
  %3 = linalg.matmul ins(%0, %1 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %2 : tensor<256x256xf32>
}
