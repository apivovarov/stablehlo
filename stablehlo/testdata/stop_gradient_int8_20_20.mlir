// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xi8>
    %1 = call @expected() : () -> tensor<20x20xi8>
    stablehlo.custom_call @check.expect_eq(%0, %1) {has_side_effect = true} : (tensor<20x20xi8>, tensor<20x20xi8>) -> ()
    return %0 : tensor<20x20xi8>
  }
  func.func private @inputs() -> (tensor<20x20xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000500FF000003F90301FE01020201FFFD02FFF906FDFF0103010000010300FB01FC0100FEFF0402FF00FD04FD01FE01FD0001FF0100FDFC0001FE00FF02FE04FCFF0000010406FE0000000002FA0000FE0100FFFF00FCFE00FF000200FF010202FD0004000500FD01000100FAFE01FC030001040102F9000201FFFB0002040500FFFE000300FEFCFF00FEFF0003000002FD000404FE00010103FFFC03FC00FFFDFF0000FEFD0000FF03FE050000020101FEFFFEFE05FE020000FF00FE000000FFFF030100F9FF00FEF9FF01FC00FBFD01FF0002040003FEFC01FEFFFD02FEFF00FF0100FD040000FCFEFB00FE0103FDFFFC00020104FDFD00020003000300FE040000FEFE020900010100FF01FEFE00020100FEFCFF000200FF0003FA00FF00FFFEFE040001FB0000FE0203FC020101000000FE00000300020002FE060004FE0200FC02030305FD04FD0002030203FD000400FF00FB0603030100000002FF0301FD000302010104FF00FAFE04FF03FCFE00020104FF00FB020103020100FDFE00FE0102FA0100FEFE0100FF00000500"> : tensor<20x20xi8>
    return %c : tensor<20x20xi8>
  }
  func.func private @expected() -> (tensor<20x20xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000500FF000003F90301FE01020201FFFD02FFF906FDFF0103010000010300FB01FC0100FEFF0402FF00FD04FD01FE01FD0001FF0100FDFC0001FE00FF02FE04FCFF0000010406FE0000000002FA0000FE0100FFFF00FCFE00FF000200FF010202FD0004000500FD01000100FAFE01FC030001040102F9000201FFFB0002040500FFFE000300FEFCFF00FEFF0003000002FD000404FE00010103FFFC03FC00FFFDFF0000FEFD0000FF03FE050000020101FEFFFEFE05FE020000FF00FE000000FFFF030100F9FF00FEF9FF01FC00FBFD01FF0002040003FEFC01FEFFFD02FEFF00FF0100FD040000FCFEFB00FE0103FDFFFC00020104FDFD00020003000300FE040000FEFE020900010100FF01FEFE00020100FEFCFF000200FF0003FA00FF00FFFEFE040001FB0000FE0203FC020101000000FE00000300020002FE060004FE0200FC02030305FD04FD0002030203FD000400FF00FB0603030100000002FF0301FD000302010104FF00FAFE04FF03FCFE00020104FF00FB020103020100FDFE00FE0102FA0100FEFE0100FF00000500"> : tensor<20x20xi8>
    return %c : tensor<20x20xi8>
  }
}
