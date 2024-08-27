// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xi64>, tensor<2x3xi64>, tensor<2x3xi64>)
    %1 = call @expected() : () -> tensor<2x3xi64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xi64>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xi64>
    stablehlo.custom_call @check.expect_eq(%7, %1) {has_side_effect = true} : (tensor<2x3xi64>, tensor<2x3xi64>) -> ()
    return %7 : tensor<2x3xi64>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xi64> {mhlo.layout_mode = "default"}, tensor<2x3xi64> {mhlo.layout_mode = "default"}, tensor<2x3xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 0], [1, 1, 0]]> : tensor<2x3xi32>
    %c_0 = stablehlo.constant dense<[[1, 2, -1], [0, -3, -3]]> : tensor<2x3xi64>
    %c_1 = stablehlo.constant dense<[[6, -2, -3], [0, -4, -1]]> : tensor<2x3xi64>
    %c_2 = stablehlo.constant dense<[[1, 2, -2], [0, -2, 5]]> : tensor<2x3xi64>
    return %c, %c_0, %c_1, %c_2 : tensor<2x3xi32>, tensor<2x3xi64>, tensor<2x3xi64>, tensor<2x3xi64>
  }
  func.func private @expected() -> (tensor<2x3xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 2, -1], [0, -4, -3]]> : tensor<2x3xi64>
    return %c : tensor<2x3xi64>
  }
}
