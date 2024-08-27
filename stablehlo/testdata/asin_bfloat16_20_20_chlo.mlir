// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = chlo.asin %0 : tensor<20x20xbf16> -> tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %2 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xDCC0A03F3AC0B9BEB9BF96C085C04A40D2BF5BC0C63F8AC0AF3F2EC08C3FE83F5ABEC7400E40783F6DBF57C086C01E40893F50BF8FC05AC0A0C09BBF5FBFBABE5C405CBFFD3F8E4026C024404EBF20C004C080BF3E407D3F283F523F8840E2BFCEBFE2BF48C095BF4BC029C0ADC06D402CBF93BFF9BFCFBFFEBF43C0C7BF6CBF95C03740823F2A3F324002BF9D3F9E3F7740FBBEF13EA6BF95C0814093408C4004C056BF3CC03F40494060C090C05AC08C4006C01BC0853F67BF6B40DAC0DB400C3E0C419ABF90BFC63F1D3DB840D73F90C00FC0D2BFFDBFC43F074084C073C09F3F254027C0154060C07E4006402040ABBF56C058BE48BF9E3F8C40A03FFBBF2440E33F8D40A6BE8940E5403040A83E12BFDAC03B3F65C04E40E73F94BF4ABE08401DBF773EC53C87BEADC058C0013F5E40F2BEF3BE39408CC04E40D23F56C02F40693F9EC02840FB3F83C05E40B1C066BFDABFCCC0974008C0C0BF3E3D89BFE73FB13F924093401C3E0A4078408B40EAC0AE3EB33F90C0013D7DBFC63FB24032C0D1C031BED5C082BF343F47C067C05FBF814008C0C44052BE05C11D40CE40D8BFA63FC13EFABFB1BF07C05A3F2440804084BF61BF263E4EBF92C06AC095BFBEBE43401740C13F2040B13FDE3FD83FC5BFBA40EBBE784089BFF5BFF3BFF3BE35C076C0F73F3CC0A0C065BE0E41D13F03C082403040BE3E10BEBD4046C08F3ECFBFBB3F4DC0DE4083C03C40123E97C00640A2BFDE3EAA4069BEE7406B4071C005C0DABFEFBED73FDE3FF9BFABBFB3BD0FC005BF98C00FBF024061C018C095C0A9C0A0405FBFDABF7DBF203F56BFDBBEC13F0A40E73E3C3FD83F5940F5BFBA3F973F1EC08F3FA2C0963F5CC06B40F1BF99BF473E0F40B33FD94029C070C0AB40DB3F29C024C0F6BF85BE30C03C408EBF923E4BC0A9C0A53F9FC08C3FDDBFA53FB43F37C02B40F9BDFFBF90BEF13DDB3F344065BE09C087BE8CC0E73F0A4096C08F409440524055C05740024004C09D3FF1BF07C0DD3FF23F15BFA240CFBF89BF834087C0793E294007BE8340F6BF3F3FED3F993D14C04640E4C001401C3FB13EEC3FA540FC3EA5BF32C03BC0F73EC3401D3EA9C038BF9DC07CBE5440193F5640"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC0FFC0FFC0FFBDBEC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF5CBEC0FFC0FFA93F98BFC0FFC0FFC0FFC0FF73BFC0FFC0FFC0FFC0FF87BFBFBEC0FF84BFC0FFC0FFC0FFC0FF6FBFC0FFC0FFC9BFC0FFB53F383F763FC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF3DBFC0FFC0FFC0FFC0FFC0FFC0FF96BFC0FFC0FFC0FF3A3FC0FF09BFC0FFC0FFC0FF03BFFB3EC0FFC0FFC0FFC0FFC0FFC0FF7EBFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF90BFC0FFC0FFC0FF0C3EC0FFC0FFC0FFC0FF1D3DC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF5ABE65BFC0FFC0FFC0FFC0FFC0FFC0FFC0FFA8BEC0FFC0FFC0FFAB3E1BBFC0FF513FC0FFC0FFC0FFC0FF4BBEC0FF29BF7A3EC53C88BEC0FFC0FF083FC0FFFDBEFEBEC0FFC0FFC0FFC0FFC0FFC0FF923FC0FFC0FFC0FFC0FFC0FFC0FF8FBFC0FFC0FFC0FFC0FFC0FF3E3DC0FFC0FFC0FFC0FFC0FF1D3EC0FFC0FFC0FFC0FFB23EC0FFC0FF013DB5BFC0FFC0FFC0FFC0FF32BEC0FFC0FF483FC0FFC0FF87BFC0FFC0FFC0FF54BEC0FFC0FFC0FFC0FFC0FFC63EC0FFC0FFC0FF823FC0FFC0FFC0FF89BF273E6FBFC0FFC0FFC0FFC3BEC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFF4BEC0FFC0FFC0FFC0FFFEBEC0FFC0FFC0FFC0FFC0FF67BEC0FFC0FFC0FFC0FFC0FFC33E11BEC0FFC0FF913EC0FFC0FFC0FFC0FFC0FFC0FF133EC0FFC0FFC0FFE53EC0FF6CBEC0FFC0FFC0FFC0FFC0FFF9BEC0FFC0FFC0FFC0FFB3BDC0FF0CBFC0FF18BFC0FFC0FFC0FFC0FFC0FFC0FF87BFC0FFB5BF2D3F7EBFE2BEC0FFC0FFF03E533FC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF483EC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF86BEC0FFC0FFC0FF943EC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFFABDC0FF92BEF23DC0FFC0FF67BEC0FF88BEC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FFC0FF1FBFC0FFC0FFC0FFC0FFC0FF7C3EC0FF08BEC0FFC0FF573FC0FF993DC0FFC0FFC0FFC0FF273FB53EC0FFC0FF033FC0FFC0FFC0FF013FC0FF1E3EC0FF4DBFC0FF7FBEC0FF243FC0FF"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
}
