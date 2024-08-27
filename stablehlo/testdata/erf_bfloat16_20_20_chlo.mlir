// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = chlo.erf %0 : tensor<20x20xbf16> -> tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %2 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x70BF7FBF00C02ABF0A414240E3BFB5BD113FF53F02C0DB3FE23FB1400ABF01C0D13D064060C0FF3F7DC04740E93FC0C06D401A4018BF2640BE3F4BBF7E40483FFFBFCA3FB7C09ABE29C0C03FDDBF024071BF853FA1C05B40433F6FC0D0BFF23FB83E7BC025C0993C8240AC40A04067BF9C3F753F85406F40BAC0BDBF714043C0E44013C073BDED3FC73E9E40D33DB53F18400A4037C081400CC01640FF3FD9BF60BF5CC0413E52C000C0A4BF01C009C014BFC9BE2440373F92BF85C0CF3FFCBDE73B463F743F233FD1BE3F4058BF853FC5BC54C0ACBE9BBFA0408CC081C0E9BF4ABF8AC0764034BF04403140464001C01AC03140C53DE4BF52401BC0513E734009C02AC08EC07E3C373F113E5EBF43BFA2BFF1BF343F0C3F5A40C9BE6AC003C07A40D7BF3A3E97C08DC0934003BF69BE153FD43F6340C53F50C085C0A74094BD5D40CE40213F323F17BF73BFE2C08E3F7A40AA3F4940CABF94BF2CC029BEC93F2C40B34095C0D7BF4BBFA0C08EBB393EDDBFC43F8B4055BFA0408940683CBF3F7240EB40CEBE1DC053C0C03F934049BEE3BFE3BFC83F47C0733F3540A7BF07C06FC07CC074BFEB3FF13F69BE373FC6C051BFFF3F59C016C08740B6C0B0BF173FB6BEBDBFDA3F104004C0EEBF5BBE9FBF433FDFBF4B40FF3F13C023BFC8404F4083BE023DD73F454046C0C4C073C0BEC0CBBEC13FA5C0A5BEBE3FA340EE3F20BF5F3E60408A4007C0A23ECEBF753D74BF2340EEBF5FC0A6C0B0BE4540A93E873F99C0AEC0BAC01A40E1BF463ED73D11C0B540BA40484079BF10BF9AC0AC40BCBFBBBE0E405F4055C0D0BEC34024C05AC0AC3F174095BF48401FC086C07B3F8ABFAAC0403FD33FB5BF2CC033402FC015BEF83F7D406F406E40933E34C04C4082BE73C0973F0A3E7DC04240EF3EF3BF373F8DBFDE40AF3FBA40803F2DC02BC09A40F8BF12409DC00EC08CBE153F28C0B3BF29C018BF30C012C0383E9CBF6EC0F93FE03FADBFE23FD34002C0694098BF32C0A6C0E4400240B3C028C009BE073F81C042C0BA3F4ABE613E08406A400CBF32405840E03FB6C02FBF4EC01CC0CEBF8540AD40A73F00C0CFBF843F864040C007C09E3FE5C0983FD8BE18C09840F4BFEA40"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x51BF57BF7FBF27BF803F803F7DBFCCBD143F7E3F7FBF7C3F7D3F803F0EBF7FBFEB3D7F3F80BF7F3F80BF803F7D3F80BF803F803F19BF803F773F3DBF803F3B3F7FBF793F80BFA9BE80BF773F7CBF7F3F51BF5C3F80BF803F383F80BF7ABF7E3FC73E80BF80BFAD3C803F803F803F4CBF6A3F533F803F803F80BF77BF803F80BF803F80BF89BD7E3FD63E803FED3D743F803F7F3F80BF803F7FBF803F7F3F7CBF49BF80BF573E80BF7FBF6EBF7FBF7FBF16BFD8BE803F303F65BF80BF7A3F0DBE023C3A3F533F223FDFBE803F44BF5C3FDEBC80BFBBBE6ABF803F80BF80BF7DBF3CBF80BF803F2EBF7F3F803F803F7FBF80BF803FDE3D7DBF803F80BF693E803F7FBF80BF80BF8F3C303F233E48BF38BF6DBF7EBF2E3F103F803FD8BE80BF7FBF803F7CBF503E80BF80BF803F08BF81BE173F7B3F803F783F80BF80BF803FA7BD803F803F203F2D3F19BF52BF80BF623F803F713F803F79BF66BF80BF3DBE793F803F803F80BF7CBF3DBF80BFA0BB4F3E7CBF783F803F43BF803F803F833C773F803F803FDCBE80BF80BF773F803F60BE7DBF7DBF793F80BF523F803F6FBF7FBF80BF80BF53BF7E3F7E3F81BE303F80BF40BF7F3F80BF80BF803F80BF73BF193FC5BE77BF7C3F803F7FBF7EBF73BE6CBF383F7CBF803F7F3F80BF22BF803F803F91BE133D7C3F803F80BF80BF80BF80BFDABE783F80BFB4BE773F803F7E3F20BF783E803F803F7FBFB13E7ABF8A3D53BF803F7EBF80BF80BFBFBE803FB83E5D3F80BF80BF80BF803F7DBF5D3EF23D80BF803F803F803F55BF13BF80BF803F76BFCABE803F803F80BFDEBE803F80BF80BF713F803F66BF803F80BF80BF563F5FBF80BF363F7B3F74BF80BF803F80BF27BE7E3F803F803F803FA13E80BF803F90BE80BF683F1B3E80BF803FFB3E7EBF303F61BF803F723F803F583F80BF80BF803F7EBF803F80BF80BF9ABE173F80BF74BF80BF19BF80BF80BF4D3E6ABF80BF7E3F7D3F72BF7D3F803F7FBF803F68BF80BF80BF803F7F3F80BF80BF1ABE0B3F80BF80BF763F61BE7A3E7F3F803F10BF803F803F7D3F80BF2BBF80BF80BF7ABF803F803F6F3F7FBF7ABF5B3F803F80BF7FBF6B3F80BF683FE6BE80BF803F7EBF803F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
}
