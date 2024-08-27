// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3xf16>, tensor<3xf16>)
    %1 = call @expected() : () -> tensor<3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<3xf16>, tensor<3x1xi64>, tensor<3xf16>) -> tensor<3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf16>, tensor<3xf16>) -> ()
    return %2 : tensor<3xf16>
  }
  func.func private @inputs() -> (tensor<3xf16> {mhlo.layout_mode = "default"}, tensor<3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[5.929690e+00, -2.326170e+00, -3.642580e-01]> : tensor<3xf16>
    %cst_0 = stablehlo.constant dense<[-4.289060e+00, 4.214840e+00, -3.269530e+00]> : tensor<3xf16>
    return %cst, %cst_0 : tensor<3xf16>, tensor<3xf16>
  }
  func.func private @expected() -> (tensor<3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.500000e+01, -3.262500e+01, -3.642580e-01]> : tensor<3xf16>
    return %cst : tensor<3xf16>
  }
}
