// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>)
    %1 = call @expected() : () -> tensor<4x2x3x5xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x2x3x5xf32>, tensor<2xi64>, tensor<4x3xf32>) -> tensor<4x2x3x5xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xf32>, tensor<4x2x3x5xf32>) -> ()
    return %2 : tensor<4x2x3x5xf32>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf32> {mhlo.layout_mode = "default"}, tensor<4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x7CB29B40787202BF00731E3D845D99BF07B611BFDD83F6BF27BDB5BFB902463E52C0853FA7F001BF3CC53540696496BFE9A1EDBF4C1FABC021CDFF3E521318401D0A1DC0A87FE940F6D74A3FC1F4AF3FEFB75440290263C03EBC9D3F456784C071AA044049853EBFDB0FABBFF44064BE292E213FCCA2C74064A52ABFAEBB01C0BE7FAA40D84C06BFFFE3F3BF268766BF85539EBD251F85C0791BD9BE024CA2C00A055CBF70AD3B3FBDDB7BC0D1D347C09BAF94BF798F67408E1A85C0D425DFC0E84DC3BF0E6B8E3F32AF3540D801F43FBEBD57BEF72E70C0E60F4A402668B0C0B9BCD3C00C076640C19822C09B73DEBFE77EECBF6BB9CB40D5BD3240206286C032BA16405C693A40C82A58BF557981BE7D5B394063D10640DFBD8DBD6C558C3E1C70E33EF7A354C0309D1AC04E003E406AD74CC04F23103CADF7C43F8BBA3AC0F49D993D435BD1BF3A2DB640206ADD3F08ACA8C0B670D5BF1A4870BF22C3AB3E1A7C4FC047FC99407BCC69C063A998BE66A5DE3DAAA3D0BF78F01A40D1370EC09FCC9EBF48620EC002E45DC0061A943F1E14EFBCAD8884C0EBA897C019B2A140607949408CC83D3F4476EABD214FF2BF409DC13FD9D8AC3F1CEA45C0A5B02340D45992BF53CB163F33F623C0548388BF75C09CC056AA72403D89FEBF226AC8BF"> : tensor<4x2x3x5xf32>
    %cst_0 = stablehlo.constant dense<[[-8.52962398, 0.358306259, 4.15123272], [1.51394248, 0.399344951, -4.01372528], [-6.03845072, 4.46726418, -0.0715741366], [0.290659636, 1.07122183, 1.37316954]]> : tensor<4x3xf32>
    return %cst, %cst_0 : tensor<4x2x3x5xf32>, tensor<4x3xf32>
  }
  func.func private @expected() -> (tensor<4x2x3x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x7CB29B40787202BF00731E3D845D99BF07B611BFDD83F6BF27BDB5BFB902463E52C0853FEB73B73E3CC53540696496BFE9A1EDBF4C1FABC0E6D68440521318401D0A1DC0A87FE940F6D74A3FC1F4AF3FEFB75440290263C03EBC9D3F456784C071AA044049853EBFDB0FABBFF44064BE292E213FCCA2C74064A52ABFAEBB01C0BE7FAA40D84C06BFDEC8C13F268766BF85539EBD251F85C0791BD9BEF176CC3E0A055CBF70AD3B3FBDDB7BC0D1D347C09BAF94BF798F67408E1A85C0D425DFC0E84DC3BF0E6B8E3F32AF3540D801F43FBEBD57BEF72E70C0E60F4A402668B0C0B9BCD3C00C076640C19822C09B73DEBFE77EECBF6BB9CB40D5BD3240206286C032BA16405C693A40C82A58BF557981BE7D5B3940D4F38E40DFBD8DBD6C558C3E1C70E33EF7A354C0769592BD4E003E406AD74CC04F23103CADF7C43F8BBA3AC0F49D993D435BD1BF3A2DB640206ADD3F08ACA8C0B670D5BF1A4870BF22C3AB3E1A7C4FC047FC99407BCC69C063A998BE66A5DE3DAAA3D0BF78F01A40D1370EC09FCC9EBF48620EC002E45DC0061A943F1E14EFBCAD8884C0EBA897C019B2A140607949408CC83D3F4476EABD214FF2BF409DC13FD9D8AC3F1CEA45C0A5B02340D45992BF53CB163F33F623C0548388BF75C09CC056AA72403D89FEBF226AC8BF"> : tensor<4x2x3x5xf32>
    return %cst : tensor<4x2x3x5xf32>
  }
}
