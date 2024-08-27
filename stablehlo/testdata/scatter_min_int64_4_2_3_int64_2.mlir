// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xi64>, tensor<2xi64>)
    %1 = call @expected() : () -> tensor<4x2x3xi64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i64>
      stablehlo.return %3 : tensor<i64>
    }) : (tensor<4x2x3xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x2x3xi64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3xi64>, tensor<4x2x3xi64>) -> ()
    return %2 : tensor<4x2x3xi64>
  }
  func.func private @inputs() -> (tensor<4x2x3xi64> {mhlo.layout_mode = "default"}, tensor<2xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[2, -5, 0], [-3, 3, -1]], [[2, 0, -6], [4, 4, 4]], [[0, -3, -4], [-4, 2, -1]], [[2, 0, -1], [-2, -3, 0]]]> : tensor<4x2x3xi64>
    %c_0 = stablehlo.constant dense<-1> : tensor<2xi64>
    return %c, %c_0 : tensor<4x2x3xi64>, tensor<2xi64>
  }
  func.func private @expected() -> (tensor<4x2x3xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[2, -5, 0], [-3, 3, -1]], [[2, 0, -6], [4, 4, 4]], [[0, -3, -4], [-4, 2, -1]], [[2, 0, -1], [-2, -3, -1]]]> : tensor<4x2x3xi64>
    return %c : tensor<4x2x3xi64>
  }
}
