// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xi32>, tensor<20x20xi32>)
    %1 = call @expected() : () -> tensor<20x20xi32>
    %2 = stablehlo.xor %0#0, %0#1 : tensor<20x20xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<20x20xi32>, tensor<20x20xi32>) -> ()
    return %2 : tensor<20x20xi32>
  }
  func.func private @inputs() -> (tensor<20x20xi32> {mhlo.layout_mode = "default"}, tensor<20x20xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040000000100000002000000FDFFFFFF03000000FCFFFFFF020000000100000001000000020000000300000000000000FEFFFFFF00000000030000000300000000000000FDFFFFFF00000000FAFFFFFF00000000FDFFFFFF00000000FFFFFFFF030000000000000001000000FBFFFFFF03000000040000000500000000000000FEFFFFFFFFFFFFFFFFFFFFFF0200000003000000000000000000000000000000FEFFFFFFFCFFFFFF0200000000000000FDFFFFFF05000000FEFFFFFFFEFFFFFFFDFFFFFF00000000FEFFFFFF000000000000000001000000FAFFFFFF0100000002000000FEFFFFFF04000000040000000000000006000000FFFFFFFFFEFFFFFF02000000FEFFFFFFFCFFFFFFFBFFFFFFFDFFFFFF03000000FEFFFFFFFFFFFFFF0000000002000000FFFFFFFFFEFFFFFF00000000FEFFFFFFFFFFFFFF02000000FFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFF0000000002000000FEFFFFFF00000000FEFFFFFF03000000FBFFFFFF06000000FBFFFFFF030000000600000002000000FFFFFFFF0200000000000000F9FFFFFF000000000000000000000000FFFFFFFFFFFFFFFFFCFFFFFF0100000001000000FEFFFFFF01000000FDFFFFFF02000000FFFFFFFF04000000FFFFFFFFFBFFFFFF00000000FDFFFFFF03000000FBFFFFFF0200000000000000FEFFFFFF040000000000000002000000FAFFFFFF00000000FCFFFFFF00000000FEFFFFFFFEFFFFFF010000000300000004000000000000000000000006000000FBFFFFFFFFFFFFFFFEFFFFFFFDFFFFFF00000000FCFFFFFFFCFFFFFF01000000FFFFFFFFFFFFFFFF0000000003000000FFFFFFFFFFFFFFFF0200000000000000FDFFFFFF0400000003000000F8FFFFFF01000000FAFFFFFFFDFFFFFF00000000010000000300000004000000F9FFFFFF0000000000000000010000000300000002000000FFFFFFFFFDFFFFFFFAFFFFFF040000000000000000000000010000000400000000000000FDFFFFFFF9FFFFFFFDFFFFFF01000000FEFFFFFFFDFFFFFF0200000000000000FFFFFFFF0000000002000000020000000300000002000000FDFFFFFF01000000020000000000000000000000FBFFFFFF0200000003000000FFFFFFFF010000000000000003000000FFFFFFFF0000000002000000FFFFFFFF04000000FBFFFFFFFFFFFFFF00000000FCFFFFFF00000000FEFFFFFF0300000004000000F9FFFFFF00000000000000000200000000000000FAFFFFFF01000000FDFFFFFF030000000200000000000000010000000000000000000000FEFFFFFF0100000000000000FEFFFFFF01000000FFFFFFFF01000000FDFFFFFF01000000FFFFFFFFF9FFFFFF00000000010000000100000000000000030000000300000000000000FDFFFFFFFBFFFFFF00000000FCFFFFFFFCFFFFFFFFFFFFFFFFFFFFFF000000000600000000000000000000000400000000000000FBFFFFFF07000000010000000000000001000000FFFFFFFFFFFFFFFF0100000000000000FCFFFFFFFFFFFFFF000000000400000004000000FFFFFFFFFEFFFFFF04000000FEFFFFFF00000000FEFFFFFF01000000010000000200000002000000FFFFFFFFFCFFFFFF0300000002000000FEFFFFFF0000000000000000FEFFFFFFFFFFFFFF00000000FFFFFFFFFEFFFFFF00000000FDFFFFFFFFFFFFFF00000000FBFFFFFF0500000002000000FFFFFFFFFCFFFFFF0000000002000000FEFFFFFF0000000000000000FEFFFFFF020000000100000002000000FDFFFFFF000000000000000000000000FFFFFFFFFFFFFFFFFDFFFFFFFEFFFFFF0000000001000000FFFFFFFF0100000000000000FEFFFFFF000000000400000000000000FEFFFFFFFFFFFFFF03000000FEFFFFFF04000000FDFFFFFF0000000000000000010000000000000002000000FEFFFFFF01000000F9FFFFFFFCFFFFFFFFFFFFFFF7FFFFFF07000000FEFFFFFFFFFFFFFF0000000001000000060000000000000003000000000000000000000002000000FBFFFFFFFEFFFFFF00000000010000000000000000000000FCFFFFFFFDFFFFFF03000000020000000000000000000000FEFFFFFFFFFFFFFF0000000000000000FFFFFFFFFBFFFFFF020000000100000002000000010000000300000001000000FFFFFFFF04000000FFFFFFFF02000000FFFFFFFFFEFFFFFFFDFFFFFFFFFFFFFF02000000FDFFFFFFFFFFFFFF0000000000000000"> : tensor<20x20xi32>
    %c_0 = stablehlo.constant dense<"0x04000000F8FFFFFF0000000004000000020000000000000004000000FCFFFFFF0000000003000000FDFFFFFF00000000FFFFFFFF01000000FEFFFFFF0100000000000000020000000000000003000000030000000100000004000000010000000400000001000000030000000100000005000000FBFFFFFFFFFFFFFF0100000000000000FDFFFFFF0300000000000000000000000500000004000000FFFFFFFF0200000000000000FFFFFFFFFEFFFFFF00000000FDFFFFFF03000000000000000000000000000000FBFFFFFF0300000002000000FEFFFFFF000000000000000003000000FBFFFFFF03000000FEFFFFFF010000000000000000000000FFFFFFFF040000000000000007000000000000000400000000000000FCFFFFFF0100000002000000FDFFFFFF0000000000000000FFFFFFFF03000000FFFFFFFFFFFFFFFF0300000000000000FBFFFFFF01000000FDFFFFFF0500000001000000030000000200000002000000FEFFFFFF030000000500000002000000FFFFFFFF03000000FFFFFFFFFEFFFFFFFEFFFFFFFFFFFFFF010000000100000000000000FDFFFFFFFEFFFFFFFEFFFFFF0100000000000000000000000200000000000000FCFFFFFFFDFFFFFF0100000000000000FCFFFFFF020000000000000000000000000000000600000003000000F9FFFFFF020000000000000000000000FEFFFFFF02000000FDFFFFFF01000000FFFFFFFF0100000003000000FBFFFFFF00000000FCFFFFFF0000000003000000000000000300000001000000000000000000000003000000FEFFFFFF0000000001000000FFFFFFFF00000000FFFFFFFFFFFFFFFFFDFFFFFFFBFFFFFFFDFFFFFFFFFFFFFF010000000300000004000000000000000100000004000000FDFFFFFFFBFFFFFF0200000000000000FEFFFFFFFFFFFFFF00000000FDFFFFFFFDFFFFFFFEFFFFFF00000000FEFFFFFF00000000FEFFFFFF010000000200000008000000010000000200000004000000000000000000000000000000FCFFFFFF02000000FFFFFFFFFCFFFFFFFEFFFFFF0300000000000000FEFFFFFF01000000010000000400000000000000FFFFFFFF00000000050000000000000000000000FFFFFFFF01000000FFFFFFFF0100000002000000FFFFFFFF0100000002000000FEFFFFFFFDFFFFFF01000000FDFFFFFF0100000002000000FFFFFFFF05000000000000000000000000000000000000000000000003000000FFFFFFFF00000000FFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000F8FFFFFFFEFFFFFF02000000FFFFFFFF00000000FFFFFFFFFDFFFFFFFDFFFFFF0200000001000000FDFFFFFFFEFFFFFF0200000000000000FDFFFFFF0100000001000000FFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFF00000000FEFFFFFFF7FFFFFFFFFFFFFF02000000020000000200000000000000030000000000000000000000FAFFFFFFFDFFFFFF01000000FDFFFFFF070000000000000003000000F8FFFFFFFFFFFFFFFEFFFFFF0300000005000000FBFFFFFF01000000FFFFFFFF00000000FFFFFFFF03000000010000000200000003000000FFFFFFFFFEFFFFFF00000000FDFFFFFFFDFFFFFFFFFFFFFFFFFFFFFF03000000FFFFFFFFFEFFFFFF0000000000000000FFFFFFFF0000000000000000FEFFFFFF000000000000000000000000010000000000000001000000000000000000000000000000000000000200000004000000FFFFFFFF01000000FFFFFFFF00000000FDFFFFFF00000000FEFFFFFFFFFFFFFFFFFFFFFF03000000FDFFFFFF000000000600000000000000FCFFFFFFFCFFFFFF00000000FCFFFFFFFFFFFFFF0000000003000000000000000000000001000000FCFFFFFF00000000FEFFFFFFFDFFFFFFFBFFFFFFFEFFFFFFFCFFFFFFFEFFFFFF00000000FFFFFFFF01000000FEFFFFFF01000000FFFFFFFF00000000FFFFFFFF040000000000000001000000FFFFFFFF03000000FAFFFFFF02000000FEFFFFFFFDFFFFFF0200000002000000030000000100000000000000FBFFFFFF0100000000000000020000000000000002000000FFFFFFFF05000000FEFFFFFFFDFFFFFF02000000FAFFFFFFFFFFFFFFFDFFFFFF00000000FBFFFFFF00000000FBFFFFFF0400000000000000FCFFFFFF030000000100000000000000FEFFFFFF00000000FEFFFFFFFFFFFFFFFDFFFFFF010000000000000003000000FFFFFFFF"> : tensor<20x20xi32>
    return %c, %c_0 : tensor<20x20xi32>, tensor<20x20xi32>
  }
  func.func private @expected() -> (tensor<20x20xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000000F9FFFFFF02000000F9FFFFFF01000000FCFFFFFF06000000FDFFFFFF0100000001000000FEFFFFFF000000000100000001000000FDFFFFFF0200000000000000FFFFFFFF00000000F9FFFFFF03000000FCFFFFFF04000000FEFFFFFF070000000100000002000000FAFFFFFF06000000FFFFFFFFFAFFFFFF01000000FEFFFFFF02000000FCFFFFFF02000000030000000500000004000000FFFFFFFFFCFFFFFFFCFFFFFFFDFFFFFFFEFFFFFFFDFFFFFFF8FFFFFFFDFFFFFFFEFFFFFFFDFFFFFF00000000050000000300000002000000FFFFFFFFFAFFFFFF01000000010000000500000007000000FAFFFFFF0100000006000000FFFFFFFF0100000006000000FEFFFFFFFBFFFFFFFBFFFFFFF9FFFFFF0300000002000000FEFFFFFF02000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFF00000000FDFFFFFFFCFFFFFFFFFFFFFF04000000FCFFFFFFFDFFFFFF07000000FFFFFFFF03000000FCFFFFFF010000000500000005000000FEFFFFFF01000000F9FFFFFF0100000000000000FCFFFFFFFEFFFFFF060000000100000001000000000000000200000001000000020000000000000001000000FEFFFFFF03000000FDFFFFFFFEFFFFFF0200000005000000FFFFFFFF0700000002000000FDFFFFFF03000000FBFFFFFF0400000003000000070000000600000000000000020000000400000002000000010000000100000001000000FFFFFFFF02000000F8FFFFFF04000000FCFFFFFF0000000005000000FBFFFFFFFCFFFFFFFFFFFFFFFDFFFFFF00000000FFFFFFFF0200000001000000FEFFFFFF0000000000000000FCFFFFFF0000000002000000F9FFFFFFFDFFFFFF020000000500000000000000FCFFFFFF01000000FBFFFFFFF9FFFFFFFDFFFFFFFAFFFFFF010000000400000007000000FFFFFFFF00000000FCFFFFFFFEFFFFFFFCFFFFFFFFFFFFFF03000000FAFFFFFFFAFFFFFF0100000002000000090000000500000002000000F9FFFFFFF9FFFFFFFDFFFFFF0100000002000000FFFFFFFFFDFFFFFFFCFFFFFF010000000300000002000000FCFFFFFF0200000003000000F9FFFFFF01000000FDFFFFFF0000000005000000FBFFFFFF02000000FCFFFFFFFEFFFFFFFEFFFFFF010000000100000000000000010000000000000001000000F9FFFFFFFAFFFFFF0200000001000000FEFFFFFFFFFFFFFFFBFFFFFF0300000004000000F9FFFFFF000000000000000001000000FFFFFFFFFAFFFFFFFEFFFFFF03000000FCFFFFFFFDFFFFFFFFFFFFFFFEFFFFFF00000000F8FFFFFF0000000003000000FFFFFFFFFEFFFFFFFEFFFFFF02000000FCFFFFFFFFFFFFFF0000000002000000070000000200000001000000FCFFFFFF0100000002000000FCFFFFFFFFFFFFFF020000000700000000000000020000000B00000000000000FDFFFFFF02000000040000000000000003000000040000000000000001000000FAFFFFFF00000000FDFFFFFF06000000FFFFFFFFFCFFFFFFF9FFFFFFFFFFFFFF02000000FCFFFFFF05000000FFFFFFFF0500000000000000FEFFFFFFFBFFFFFFFDFFFFFF01000000FCFFFFFF02000000FEFFFFFFFCFFFFFF020000000200000001000000FCFFFFFFFDFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFEFFFFFFFDFFFFFFFFFFFFFF00000000FAFFFFFF0500000003000000FFFFFFFFFCFFFFFF0000000002000000FCFFFFFF04000000FFFFFFFFFFFFFFFFFDFFFFFF01000000FFFFFFFFFDFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFCFFFFFF02000000FDFFFFFFF8FFFFFF00000000FDFFFFFF0300000001000000FCFFFFFF01000000000000000700000000000000FEFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFAFFFFFF00000000FBFFFFFFFEFFFFFFFDFFFFFFFEFFFFFF02000000010000000000000007000000FDFFFFFF00000000F7FFFFFFF8FFFFFFFAFFFFFFFFFFFFFF01000000FEFFFFFF05000000FAFFFFFF01000000FEFFFFFFFDFFFFFF00000000F9FFFFFFFDFFFFFF0100000001000000FBFFFFFF01000000FCFFFFFFFFFFFFFF0300000000000000FFFFFFFF05000000000000000200000002000000FAFFFFFF000000000600000002000000FAFFFFFF02000000FAFFFFFF07000000010000000300000007000000FEFFFFFF0200000001000000FEFFFFFF0300000000000000FFFFFFFFFCFFFFFFFFFFFFFF03000000FFFFFFFF"> : tensor<20x20xi32>
    return %c : tensor<20x20xi32>
  }
}
