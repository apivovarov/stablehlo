// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xf64>, tensor<4x3xf64>)
    %1 = call @expected() : () -> tensor<4x2x3x5xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<4x2x3x5xf64>, tensor<2xi64>, tensor<4x3xf64>) -> tensor<4x2x3x5xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xf64>, tensor<4x2x3x5xf64>) -> ()
    return %2 : tensor<4x2x3x5xf64>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf64> {mhlo.layout_mode = "default"}, tensor<4x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x9E9372F790D412C0A8797BF65A700CC0F141D07685400D40F8772CCEB689E8BF4AC5FEDE3EE6F8BF568A52996D96EA3FF4CFEE30A739FBBF875CD96E4C6014C01485916EFE42B9BF9E0D33A91D530440C0BA3ED1893310C05C1EB44EE99D1D40420FBF065016F43F1C1227AD623B06408678A731ABA0FBBF56BA80D63DDCE8BF347F52C8AF28144018DCB918E9C7F43F1C48DE58BB0D0840456139B0BDE40B407E41D13EAF3C03C01E21BFEE40D4933F0A60C2CACF7C0B40E840D59E2D91F43F241D64043B7909C0DBE6B54133921240ACF067F03D5DF33FBE44434D336F1BC028F672B772F90440B06B905E0FD4F03FF3D30B88930E1340A9247B1669BDECBF33052E66695217401E4CCB7F690C0D40D902B4BCCE62E7BF42F567ABB66310C087F32CE8E04C10C0603DE53231E402C01BB901DFDFF301C0B226FAA0514B084012E8B7B6FDAD04C0C0E0802E815812C06723B64D524FFEBF9479905CF3C6E8BFAE1BB79BBC8E1E40FE94BEEA3F81EABFAB052FC2981900C09A23AE856348FCBF464934CB8C991D40347450A3A71106405047ACD9A63506406EA4B92E174C0A40C363D307D613D63FB9F503C1749D01C0442FF5B125C30340630F57D953F1DDBF712BAF7015C30F4037E08E081964144018E6B780FEDFBABFC4B9E1BDF37B0D400A5392CAD91A10C02E8B56CC29D20040A6E1D845FAD50F407BBEC080A5B6EF3F97673246C5D7F4BF68F9773E5434F03F5AAE8E3C6C19EBBFD235F00427FF08407891D861A48EE83FAE2253F13A70F73F403ADE1156E205408AE8FB913B3502C0F1FC8AD663E80040D7DB4A12F67309407097B4F7F76E0B4023C858EFC8070040008A438C5022F93FB490CCE15471FE3F5DEB08ECD1D8054010FFDBF08DC8F3BFD4B443DF854DFB3F02A7DA2D584C9ABF2819568C12871540AC874A5FF2EB01C0E85E431DBB510C40105E1592595710C053B5DD8C62861240EFD57A0789AFE23FA9341822B800F73F7D1A2DB5718601403428608058870440E241D9B3C6D0F03F7104E2F428F9F3BFAF16AD394E00FA3FB49F2B4DD1E203403AEBCA1CEA8CF6BF6C1177A11C9BF0BFD6E46D9B580E0AC0985BF47C9C3B18C021D713C1E97180BF56A0150CB8AEEFBF4A5286AC612C0CC0FC855846F7E803405CB04B40E240FC3F616D01C9C3F3F3BF9FEB2C73D846FF3F02D5342591F601407E4F06593924E23F622B3838E605EBBFC071A399B668FCBFEEBACAEE3DE0FD3FC0F09F64683804C09403C73F3F9C0C402D0D7D9FF20BEABF1E697EDADA4307C0374086E69502F03F1B34A01AE6C5124035044CFC5B8BD1BF16970D1C5E7E11C05603276C7C8CEEBF"> : tensor<4x2x3x5xf64>
    %cst_0 = stablehlo.constant dense<[[3.3836391557408705, 1.2487981856167674, -1.9592337298464826], [-1.7479993258960689, 0.78091628862510775, 1.0088126373598902], [2.737929798720689, 2.4620828551870546, -3.5910363384767292], [4.0965127893576518, 1.2913188014104597, 0.87515705706569546]]> : tensor<4x3xf64>
    return %cst, %cst_0 : tensor<4x2x3x5xf64>, tensor<4x3xf64>
  }
  func.func private @expected() -> (tensor<4x2x3x5xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x9E9372F790D412C0A8797BF65A700CC0F141D07685400D40F8772CCEB689E8BFB9ACE867091015C0568A52996D96EA3FF4CFEE30A739FBBF875CD96E4C6014C01485916EFE42B9BF9DF62043A4610940C0BA3ED1893310C05C1EB44EE99D1D40420FBF065016F43F1C1227AD623B0640F44AAB6C81100B4056BA80D63DDCE8BF347F52C8AF28144018DCB918E9C7F43F1C48DE58BB0D0840456139B0BDE40B407E41D13EAF3C03C01E21BFEE40D4933F0A60C2CACF7C0B40E840D59E2D91F43F241D64043B7909C0DBE6B54133921240ACF067F03D5DF33FBE44434D336F1BC028F672B772F90440B06B905E0FD4F03FF3D30B88930E1340A9247B1669BDECBF33052E66695217401E4CCB7F690C0D4000A673C17770F43F42F567ABB66310C087F32CE8E04C10C0603DE53231E402C01BB901DFDFF301C099849574C4F8024012E8B7B6FDAD04C0C0E0802E815812C06723B64D524FFEBF9479905CF3C6E8BF1A593AFAACD31E40FE94BEEA3F81EABFAB052FC2981900C09A23AE856348FCBF464934CB8C991D40347450A3A71106405047ACD9A63506406EA4B92E174C0A40C363D307D613D63FB9F503C1749D01C0442FF5B125C30340630F57D953F1DDBF712BAF7015C30F4037E08E081964144018E6B780FEDFBABFC4B9E1BDF37B0D400A5392CAD91A10C02E8B56CC29D20040A6E1D845FAD50F407BBEC080A5B6EF3F606AFA8F7B880CC068F9773E5434F03F5AAE8E3C6C19EBBFD235F00427FF08407891D861A48EE83F3695933188DA0C40403ADE1156E205408AE8FB913B3502C0F1FC8AD663E80040D7DB4A12F67309401B83EF12EFA028C023C858EFC8070040008A438C5022F93FB490CCE15471FE3F5DEB08ECD1D8054010FFDBF08DC8F3BFD4B443DF854DFB3F02A7DA2D584C9ABF2819568C12871540AC874A5FF2EB01C0E85E431DBB510C40105E1592595710C053B5DD8C62861240EFD57A0789AFE23FA9341822B800F73F7D1A2DB5718601403428608058870440E241D9B3C6D0F03F7104E2F428F9F3BFAF16AD394E00FA3FBF1E975BA65D24403AEBCA1CEA8CF6BF6C1177A11C9BF0BFD6E46D9B580E0AC0985BF47C9C3B18C09D656606573C85BF56A0150CB8AEEFBF4A5286AC612C0CC0FC855846F7E803405CB04B40E240FC3F70EA8CAD1876F1BF9FEB2C73D846FF3F02D5342591F601407E4F06593924E23F622B3838E605EBBFC071A399B668FCBFEEBACAEE3DE0FD3FC0F09F64683804C09403C73F3F9C0C402D0D7D9FF20BEABF1E697EDADA4307C0374086E69502F03F1B34A01AE6C5124035044CFC5B8BD1BF16970D1C5E7E11C05603276C7C8CEEBF"> : tensor<4x2x3x5xf64>
    return %cst : tensor<4x2x3x5xf64>
  }
}
