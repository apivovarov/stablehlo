// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xi32>, tensor<1x3xi32>)
    %1 = call @expected() : () -> tensor<1x50x3xi32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) : (tensor<1x50x3xi32>, tensor<1xi64>, tensor<1x3xi32>) -> tensor<1x50x3xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi32>, tensor<1x50x3xi32>) -> ()
    return %2 : tensor<1x50x3xi32>
  }
  func.func private @inputs() -> (tensor<1x50x3xi32> {mhlo.layout_mode = "default"}, tensor<1x3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xF8FFFFFF01000000FCFFFFFFFDFFFFFFFFFFFFFF010000000000000000000000FEFFFFFFFFFFFFFF0000000003000000FFFFFFFF00000000FFFFFFFFFCFFFFFF00000000FAFFFFFFFFFFFFFFFEFFFFFF000000000300000007000000FBFFFFFFFEFFFFFF0100000000000000FDFFFFFF00000000FDFFFFFF0000000002000000FBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000FFFFFFFF0200000000000000FEFFFFFF0400000003000000010000000000000000000000010000000100000000000000FDFFFFFF05000000FEFFFFFFFCFFFFFF0300000002000000FCFFFFFF01000000FFFFFFFFFDFFFFFFFDFFFFFF00000000FDFFFFFFFFFFFFFF00000000FFFFFFFFFCFFFFFFFFFFFFFF040000000000000000000000FFFFFFFF0100000004000000FDFFFFFF060000000000000003000000FFFFFFFF000000000200000002000000010000000200000001000000FFFFFFFF00000000FDFFFFFFFFFFFFFFFAFFFFFF0000000000000000FDFFFFFF03000000FFFFFFFF020000000000000000000000FFFFFFFF050000000200000000000000FDFFFFFFFFFFFFFF0100000004000000030000000000000002000000FFFFFFFF050000000000000005000000030000000300000000000000060000000000000002000000FFFFFFFF000000000000000002000000FEFFFFFF02000000FCFFFFFF020000000200000002000000FFFFFFFF0200000002000000FFFFFFFF01000000FDFFFFFFFBFFFFFF0500000000000000000000000000000000000000FDFFFFFF000000000500000003000000010000000300000003000000FDFFFFFFFAFFFFFF01000000"> : tensor<1x50x3xi32>
    %c_0 = stablehlo.constant dense<[[-2, 0, -2]]> : tensor<1x3xi32>
    return %c, %c_0 : tensor<1x50x3xi32>, tensor<1x3xi32>
  }
  func.func private @expected() -> (tensor<1x50x3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xF8FFFFFF01000000FCFFFFFFFDFFFFFFFFFFFFFF010000000000000000000000FEFFFFFFFFFFFFFF0000000003000000FFFFFFFF00000000FFFFFFFFFCFFFFFF00000000FAFFFFFFFFFFFFFFFEFFFFFF000000000300000007000000FBFFFFFFFEFFFFFF0100000000000000FDFFFFFF00000000FDFFFFFF0000000002000000FBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000FFFFFFFF0200000000000000FEFFFFFF0400000003000000010000000000000000000000010000000100000000000000FDFFFFFF05000000FEFFFFFFFCFFFFFF0300000002000000FCFFFFFF01000000FFFFFFFFFDFFFFFFFDFFFFFF00000000FDFFFFFFFFFFFFFF00000000FFFFFFFFFCFFFFFFFFFFFFFF040000000000000000000000FFFFFFFF0100000004000000FDFFFFFF060000000000000003000000FFFFFFFF000000000200000002000000010000000200000001000000FFFFFFFF00000000FDFFFFFFFFFFFFFFFAFFFFFF0000000000000000FDFFFFFF03000000FFFFFFFF02000000000000000000000000000000050000000200000000000000FDFFFFFFFFFFFFFF0100000004000000030000000000000002000000FFFFFFFF050000000000000005000000030000000300000000000000060000000000000002000000FFFFFFFF000000000000000002000000FEFFFFFF02000000FCFFFFFF020000000200000002000000FFFFFFFF0200000002000000FFFFFFFF01000000FDFFFFFFFBFFFFFF0500000000000000000000000000000000000000FDFFFFFF000000000500000003000000010000000300000003000000FDFFFFFFFAFFFFFF01000000"> : tensor<1x50x3xi32>
    return %c : tensor<1x50x3xi32>
  }
}
