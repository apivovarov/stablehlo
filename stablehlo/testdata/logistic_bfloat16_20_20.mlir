// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.negate %0 : tensor<20x20xbf16>
    %3 = stablehlo.exponential %2 : tensor<20x20xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<20x20xbf16>
    %5 = stablehlo.add %4, %3 : tensor<20x20xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<20x20xbf16>
    %7 = stablehlo.divide %6, %5 : tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %7 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC93F873F30400240A7BF0FC18E3FCE3D81C0DFC0C63EBFBFE43EB73F8640B83E3D3F24C02BC09FC0D53FBDBE6D40D14038BF813F37C0CE4091C0443FE2C066401FBF23C0E3BFA23F4CC096402CC0764086BF43C01B4090BD88BF2D3D94BD17403F3EA74081C0CDC019C113C09CBEC0BF6F40A73FDC3FF9BF37C03C40C7C02ABF61BE84C016BF18408540013FCBBE6D3FA1C0EFBF16C095C0A4408D4057C030C091C03D40E63F16C0AFC06A3E47C0343BC8C022404CC0B2C0B0BFA73FFEBFABC07640AEC0E0C0253D5F40EC3F14BFE0BE0BC0FE3F1040673FAE3EE83FCDBF1F3F574088BFA94013414B3F26C0313E19BEB4C040C0863F42BEA540B140D640303F1A40AB3EC5BF80BF094018BD0B40D83F57BE82BE8EBF7CBF17C092408AC0E23FA5C00E40BC3FEE3FEDBEEE3FA2C083C0CFBE02409BBFE2BF03C0F1BF2EBE9340EF4089C08540933DD13FDCC082C085BFA0BF15C0D5BF2E409FC019C00BC047BFCEBFB93FB0BF0EC070C0F9C08ABFC23F01C14A4061C0B33F484090C08A3E4F3E60C0B03F7540BB3F9EBFCDBF1C40133F9C40E43E15404140524068C02F415FBF104030BF1FBE943F7840E2BF5840BA40A5403740713FAD3D4C3FF2BFEABF3FC00340474169BE0D4033409BBEA1BF7DC030BF04400F40A8C017C094400E402E4097C0173F1240FB3F914004C00EC09D40C2400A3FEFBF4DBF12C10240F3BF9A3FA3404CBFBE3F2A406DBF83C0B83FB73F91BF5540DDBFEDBF8440FA3E8940C1BF8840163F2240603F6EC010BFA63FE0BF0640E9C0E2C08640333F6EBFA6C096C019C08040B43EC0C09D3F6FC028C02940C0BFABC022C081BF043F563AC73FB7C0194089C035C0B940883FA8BFA740873ECEC088BFA7BF71C0BB401341863EC23FC3C079C0ACBF904090C01DC0D0BF37404240883E7FBF80C045C00B40A63E8D40EA3F5FBFEABF04C0D1BF9E4071C04F40073FE2C0913FC4BF7ABFAFBFFF3FA9BF584080C0AFC0873E50C0FDBB2A40563D9440763E8A3F44BFD0C0A1BD0EC1B1C06FC085BFE7BF5FC00040D83F3340D93F98407CC0AFC04ABF42C0E93F0E401F40623F5CBF50C035C028C0583E8BC01E40C93F9EBF1DC077C0843F3DC07CC0953F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x533F3F3F713F623F5A3E0A39413F063F8F3C763A183F3C3E1C3F4E3F7C3F163F2D3F933D853DE23B583FD23E7A3F803FA83E3B3F5D3D803F2E3C2E3F603A783FB33E953D143E483F223D7E3F833D7A3F853E3A3D6C3FF83E843E033FF63E6A3F0C3F7E3F8F3CD93A9438BB3DD93E3B3E7A3F493F593F003E5D3D733F023BAE3EE43E833CB73E6A3F7C3F1F3FCE3E373FD53B093EB33D1B3C7E3F7C3F0A3D763D2E3C733F5C3FB33D8A3B0E3F2F3D003FFC3A6D3F223D7C3B4F3E493FF83D9C3B7A3F8E3B6F3A033F783F5D3FB83EC93ED23D603F673F363F163F5C3F2C3E263F783F843E7E3F803F303F8E3D0B3FED3E6A3B423D3D3FE73E7E3F7E3F803F2B3F6A3F153F353E8A3E653FFC3E653F583FE53EE03E7E3E8B3EB13D7E3F593C5A3FBB3B673F4F3F5D3FC53E5D3FCE3B863CCD3E623F6A3E163EEA3D073EEA3E7E3F803F5F3C7C3F053F563F873A8B3C863E643EB63D233E713FE23BAC3DD23DA13E2B3E4F3F4F3EC93DBC3CDA39823E523FA539763FEC3C4D3F753F343C113F0D3FF13C4D3F7A3F4F3F673E2C3E6C3F243F7E3F1C3F6A3F753F763FD53C803F973E673FAB3EEC3E433F7A3F163E783F803F7E3F733F383F053F303F063E0E3E453D643F803FE43E673F713FDA3E643E9B3CAB3E643F673FAB3BB13D7E3F673F713F113C253F683F603F7E3FE73DC93D7E3F803F223F093E9E3EE538623F053E453F7E3F9F3E513F6F3F913E863C4F3F4E3F783E763F1B3E0B3E7C3F1F3F7C3F393E7C3F253F6D3F353FC23CBA3E493F183E643F343A603A7C3F2B3F913EB63B163CAC3D7C3F163F223B453FC03C8A3D6F3F3B3E9C3B973D893E213F003F533F583B6A3F5F3C653D803F3F3F593E7E3F113FD23A843E5A3EB93C803F803F113F523F143BA43C553E7E3F343CA23D283E733F753F113F8A3E943C343D653F153F7C3F5C3F973E0E3EE73D273E7E3FB93C763F213F603A423F363E8C3E4F3E603F583E783F943C8A3B113F193D003F6F3F033F7E3F103F3F3FA23EC53AF63E1339823BC03C863E113EF53C623F583F713F583F7E3F9D3C8A3BA03E3C3D5C3F673F6C3F353F983E193D653D8A3D0D3F523C6C3F533F673EA23DA93C3C3F4C3D9D3C433F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
}
