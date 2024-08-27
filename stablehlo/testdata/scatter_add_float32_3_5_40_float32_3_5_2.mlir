// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x40xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x40xf32>, tensor<3x5x2xf32>)
    %1 = call @expected() : () -> tensor<3x5x40xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3x5x40xf32>, tensor<2x1xi64>, tensor<3x5x2xf32>) -> tensor<3x5x40xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x40xf32>, tensor<3x5x40xf32>) -> ()
    return %2 : tensor<3x5x40xf32>
  }
  func.func private @inputs() -> (tensor<3x5x40xf32> {mhlo.layout_mode = "default"}, tensor<3x5x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x772C6D40B80A9640947022C0F3194241EBE5F7BF6A4B20409A36ABBFB716ACBF503C10C0FA3C6D403329CB4093A0D0BF8C7F0E409CF5D9C0AB35B4BF5B9F1C3EFD2713C09CA59C3F5419274071D5DE3F9DCB4640E0B9F8BD78704840A7D1AD40F4C89440C1311540EA639FC0729B31C047284BC0040116C0B4A2244028B00840E7119D405C094240D01640BF4850B04092D01FBFD446E9BF776107C02E9C80C03351AAC0F8914C40095177C04B8446C05351E7BF1763873F5AA086406C019B3E1D5AEABF55D76DC02611B8BFD93226C0AAC32CC0394AACC09BD490C08760383F67DF48405AD5BC3F00906DBE23833F3F6EB83F4026F57ABFA40737C017485EBE0972963F3A06FA40172D3E3FA0080E403F370CC08C4B643F64C56BBFD05112403DEA8E407F9D8D403737FABF8C641640FE17D9C0A379A73EC4F671BF5D6662407B3931C0609EFC3EBB06D43F4C55D3C0564BCFC012A9173FF25694C0DC90AB3F097AEF3F0987A140A601C8BFC39937C00D4A45BFEF8A104067AE2440F8A1F8BF9890944045C9BABFA2297FC091EA114012C7634035715D40E463BB3DE31FAE4053DE983FDA8F99BF559F6C406ED8B23E0A7F6DC0CCCFE13F79CEA93FDC49E73EA39AE63E833DF9BFC053E13FC7BB33C083E942C0251EE4BD0AD4FB3EE14C49C08FC3863F70A6713FF276F8BED9F419C1C9985D40949B1340A49A30C0159588BF1C4BCC3FAAE54AC0E4E693BF7413AEC0AA3E2ABE8A80E740ABCF0440B8221AC034F4103D2A43703F88C123C0338C9C40724A6640D6E6C7BD3F6D964081EBCEBFFE0BB640CE569A3F55358E4080E3CEBE4DFB4540A7BF7BC03F639EBF17BF2E40BB0E92402B416540A5FC2C3F388820C082CD28BFD7B19AC0FC28C23FEB618740B02F9E404B5B434015610E407A8C2D402D7146C0CBABBC406650B9BF77822ABF5F34C8BECC512D40F156AABF38190A40D4BE8F3F2FF655C08EB43BC0C6E9A9400F1D66C00C4D29C04294B8C0B2DF67C06FBBB3BFE8EA94C0C915D8BFA7F1B63EF7A668C034B0DBC01D2D07BFF1D20B3F5576AC3F610BD03F5CB75540A38319BF356FB6BC37323AC0DEEAD73FA7718E4050A047BEFB29F03EE8913B400045A2BE87A109405D63A3BF5DC64CC02F1A053F1981D3BF72915DC0B655E1BF0CD3864016DE37C0233A8C3F92F85F40E6847EBF8279923F592D39BF756D97409E1A673FEBEE173F90C7C2BEDF9B4AC0CC043AC091C71E40EB11043EBA2B77C035B5D03F7B7890BE92D0FD3DF2E6E43D31620A4052EF423FBCF41240598F4140C1633AC0815D3D3F000F8E40F48C86C0BB03F93FBD409D407326F5C0746588C089CCA83F6E4AAFBF5C73A7406B2180BF95DC13C0E20BF5BE8B4D74BDE3AC0CC04067543FEBE648405E354640FC368AC08BF972BF6BDD2640493F15C0CD39BEBF3D73253F5A79B43E39F11EBFB122E7407B14B2C0B7159E3F19D81FC0B05B583F2CBA703E35AF833F7B9389BB0626FCC0A00819BF8EBC363E3E8BB13EEBF475C0489529BE69C9063F48AFC1BF5D710FC0D58CB140E415B63F01EDA140E67D8FC0A8037C40B1CCECBFDDB3074097A10F3FD8E3DA3E3E305F3F8910A6C0526C8EC0C4F5A94055545140EB8B8ABFDF891F4010E10C3E480ABCC09C9826C082B325BF1D2C4640321CA64028568FC066986CC07032943F25E0BB3F43E98D40520BB1BE6EC156C0539A8040EAB51BC0F2EB6140BC06BC40551AA03FFE9614C0C35618C1BE70FC3F8246F6BF52277A3F580BF53EDE93153FAD4D01C08DFC8240C8699C40ABABBE40641189C0AC0992C0E3BC65409B808F409A5A46409068FE3F7B911CC024E878C01E9D443FAD9752C0DC2C3A3F2996FE3F6CF41040053859C0178E1540BED93F401359833F38CAD8BFBCF6BD3D33CAB440E2CACB3F5002D53FE555D8BFBC93D3BFC2D52C40C08CE23F2DC3B2BF1D51893D29F50FBFC0747BC05223E9C02F2CA4C0FB9905C012E68C40C2F3B8C0430041408405CF3EA0678DBFFF019C40ED4C92403D6012C0034BDB3F28AAAF40C7ABD8BE7AB9C73F91E5863F0D5D0FBF76E3FCBF0720953FC7D327403EE1ABC02C94304031EBADC04C4F2DBFBCC8DA40DD32414049D97E400B3A22C00D0D31BFA544833F47090FC0DE390B40B58801C06CDDB840A2D410407B1D6B40EE9D954064619A3F58B3AE3E75E904BFE3C5CD3A822B5DC047F54DBF1374EDBFE382DBBFE4D6443F708E60C052F1BDC0846052C0822E8DC06DED3F3FFB7AD2BF14990AC0EC3953404744B43E68CAE03F680BB03F176C0040AF649C400F4EDE3F1F3BDE3F61FA93C09CB996C0E20C96C0A1AD8F40C1D20FC08BB9BEC011056BC07DA0863DBE08BEBE7A89C140AE18B43F033DBDBF1C7A6540214C48C035E74C403D2B374068AE0A406CE4FC3F2BAD39C080A39A4000E10FC0134B66BFEBA53F408D0C3CBEC7469B3E4886C0BF8BE3AAC0EDBBECBFE6BD1FC048C190BFFF78A8C01B792C3FC3234440A85683BE4DA40C403364F2BFC73E96C08C13353F085DDBC074136340C6FC1B3FB46B2240181534BEC70511C029CE66C000BDBB3FA68800BE66A0CB3F4161953EE3FC09C0BDCCC8BF8FACD3BF198BB0401F5B04BF6857AFC0CEF717C1B2A4DEBE41613A3F96D326C0EEB6084031B199BED9C744C046CA2EC088D425405CC7FA3EFBD754C04F5F81C03BE0CEBE2546D4C09E29B9C06C2D203F29B0B23F49451740A33203C0F539B43FB341A1BFE74DE23F5180A63F3BDF223FAD16CA3C3D1A08C0A835BBBFC23511BF3DD3F83F85F161BF258C943F39AF8140BB372BC0DC2944BCA8848FBFFC090CC02C925E40ECC381C045D9E23E5EB98E404B23B2BFEA8624BFCD63A54037B2AE40843AAFC012F4E7BF4FDC65BFD00FDB3F0BD088C0905A46C00325C6C099B3F13F15FA3D404513E0BF6CD9C8BD0FF49DBF4A33D93EEA5F023FE48FAF4085E98CBF63A512BFCA8B8AC0C8A46840260CBCBF94A60AC015B23EC08DD4943FC9B53F4002DC5F4073284F40C5A46B409476D4BF4A6136BF4395EF3F664CF13FF97B94408EF148C0D9241BC0E60939C065B571BDBCFD783EFA4D9140F578EC3D394D583F078C4AC0362505C08997D1C01984004014C2B7C00F86D7BFA42A0ABFC84B51C047E1DABF0D5A3CC0AFE2464061605C3F30B40BC0A8DE69C0F6A842BEE2EB9CC0CD3C96C09A93DF3F8BC815BE2A165440F2F7A040F8697EC03A150EC00B97A5BF3D1DFBBF8F5B8EC0B3F088BE939836BD8C29EE3F43E371C05BE5EE3F569FA9402D4D01BF9C8A2ABF0D6E10C03251A33F0AE037C0BA1B7BBFCC560FC0CAB27FBD443125403D932A40DFC28FC0D6E05F4056C78540085332C064F8DB3F7F2A4340CCCB82C089DF03BF"> : tensor<3x5x40xf32>
    %cst_0 = stablehlo.constant dense<[[[-1.37861621, 0.682284653], [0.844693541, -0.254867673], [-1.05782747, -1.43705845], [-0.272654504, -2.25604844], [-3.45285153, 0.28833589]], [[6.41492033, 2.985830e+00], [-4.70001554, 0.332871348], [0.0697809085, -6.60834789], [-0.451723516, -5.08355951], [-0.15722701, -1.46146595]], [[5.045030e+00, -0.588169515], [-3.00175142, 5.58849096], [6.1161108, -2.36064601], [3.44111109, 3.35898757], [-0.469992638, 1.53118742]]]> : tensor<3x5x2xf32>
    return %cst, %cst_0 : tensor<3x5x40xf32>, tensor<3x5x2xf32>
  }
  func.func private @expected() -> (tensor<3x5x40xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x772C6D40BD847F40947022C0F3194241EBE5F7BF6A4B20409A36ABBFB716ACBF503C10C0FA3C6D403329CB4093A0D0BF8C7F0E409CF5D9C0AB35B4BF5B9F1C3EFD2713C09CA59C3F5419274071D5DE3F9DCB4640E0B9F8BD78704840A7D1AD40F4C89440C1311540EA639FC0729B31C047284BC0040116C0B4A2244028B00840E7119D405C094240D01640BF4850B04092D01FBFD446E9BF776107C02E9C80C03351AAC0AE517240095177C04B8446C05351E7BF1763873F5AA086406C019B3E1D5AEABF55D76DC02611B8BFD93226C0AAC32CC0394AACC09BD490C08760383F67DF48405AD5BC3F00906DBE23833F3F6EB83F4026F57ABFA40737C017485EBE0972963F3A06FA40172D3E3FA0080E403F370CC08C4B643F64C56BBFD05112403DEA8E407F9D8D403737FABF8C641640FE17D9C0A379A73EC4F671BF5D6662407B3931C06A1800C0BB06D43F4C55D3C0564BCFC012A9173FF25694C0DC90AB3F097AEF3F0987A140A601C8BFC39937C00D4A45BFEF8A104067AE2440F8A1F8BF9890944045C9BABFA2297FC091EA114012C7634035715D40E463BB3DE31FAE4053DE983FDA8F99BF559F6C406ED8B23E0A7F6DC0CCCFE13F79CEA93FDC49E73EA39AE63E833DF9BFC053E13FC7BB33C083E942C0251EE4BD0AD4FB3EE14C49C08FC3863F52D9CABFF276F8BED9F419C1C9985D40949B1340A49A30C0159588BF1C4BCC3FAAE54AC0E4E693BF7413AEC0AA3E2ABE8A80E740ABCF0440B8221AC034F4103D2A43703F88C123C0338C9C40724A6640D6E6C7BD3F6D964081EBCEBFFE0BB640CE569A3F55358E4080E3CEBE4DFB4540A7BF7BC03F639EBF17BF2E40BB0E92402B416540A5FC2C3F388820C082CD28BFD7B19AC0FC28C23FEB618740B02F9E403484E5BD15610E407A8C2D402D7146C0CBABBC406650B9BF77822ABF5F34C8BECC512D40F156AABF38190A40D4BE8F3F2FF655C08EB43BC0C6E9A9400F1D66C00C4D29C04294B8C0B2DF67C06FBBB3BFE8EA94C0C915D8BFA7F1B63EF7A668C034B0DBC01D2D07BFF1D20B3F5576AC3F610BD03F5CB75540A38319BF356FB6BC37323AC0DEEAD73FA7718E4050A047BEFB29F03EE8913B400045A2BE87A109400EFD01415DC64CC02F1A053F1981D3BF72915DC0B655E1BF0CD3864016DE37C0233A8C3F92F85F40E6847EBF8279923F592D39BF756D97409E1A673FEBEE173F90C7C2BEDF9B4AC0CC043AC091C71E40EB11043EBA2B77C035B5D03F7B7890BE92D0FD3DF2E6E43D31620A4052EF423FBCF41240598F4140C1633AC0815D3D3F000F8E40F48C86C0BB03F93FBD409D407326F5C0746588C089CCA83F6E4AAFBFB69D5D3F6B2180BF95DC13C0E20BF5BE8B4D74BDE3AC0CC04067543FEBE648405E354640FC368AC08BF972BF6BDD2640493F15C0CD39BEBF3D73253F5A79B43E39F11EBFB122E7407B14B2C0B7159E3F19D81FC0B05B583F2CBA703E35AF833F7B9389BB0626FCC0A00819BF8EBC363E3E8BB13EEBF475C0489529BE69C9063F48AFC1BF5D710FC0D58CB140E415B63F01EDA140E67D8FC0A8037C40B1CCECBF02628DC097A10F3FD8E3DA3E3E305F3F8910A6C0526C8EC0C4F5A94055545140EB8B8ABFDF891F4010E10C3E480ABCC09C9826C082B325BF1D2C4640321CA64028568FC066986CC07032943F25E0BB3F43E98D40520BB1BE6EC156C0539A8040EAB51BC0F2EB6140BC06BC40551AA03FFE9614C0C35618C1BE70FC3F8246F6BF52277A3F580BF53EDE93153FAD4D01C08DFC8240C8699C40ABABBE40641189C05B9521C1E3BC65409B808F409A5A46409068FE3F7B911CC024E878C01E9D443FAD9752C0DC2C3A3F2996FE3F6CF41040053859C0178E1540BED93F401359833F38CAD8BFBCF6BD3D33CAB440E2CACB3F5002D53FE555D8BFBC93D3BFC2D52C40C08CE23F2DC3B2BF1D51893D29F50FBFC0747BC05223E9C02F2CA4C0FB9905C012E68C40C2F3B8C0430041408405CF3EA0678DBFFF019C40ED4C92403D6012C0E09AC13D28AAAF40C7ABD8BE7AB9C73F91E5863F0D5D0FBF76E3FCBF0720953FC7D327403EE1ABC02C94304031EBADC04C4F2DBFBCC8DA40DD32414049D97E400B3A22C00D0D31BFA544833F47090FC0DE390B40B58801C06CDDB840A2D410407B1D6B40EE9D954064619A3F58B3AE3E75E904BFE3C5CD3A822B5DC047F54DBF1374EDBFE382DBBFE4D6443F708E60C052F1BDC0846052C0822E8DC06DED3F3FB6FF334014990AC0EC3953404744B43E68CAE03F680BB03F176C0040AF649C400F4EDE3F1F3BDE3F61FA93C09CB996C0E20C96C0A1AD8F40C1D20FC08BB9BEC011056BC07DA0863DBE08BEBE7A89C140AE18B43F033DBDBF1C7A6540214C48C035E74C403D2B374068AE0A406CE4FC3F2BAD39C080A39A4000E10FC0134B66BFEBA53F408D0C3CBEC7469B3E4886C0BF8BE3AAC0EDBBECBFE6BD1FC048C190BFDA642BC01B792C3FC3234440A85683BE4DA40C403364F2BFC73E96C08C13353F085DDBC074136340C6FC1B3FB46B2240181534BEC70511C029CE66C000BDBB3FA68800BE66A0CB3F4161953EE3FC09C0BDCCC8BF8FACD3BF198BB0401F5B04BF6857AFC0CEF717C1B2A4DEBE41613A3F96D326C0EEB6084031B199BED9C744C046CA2EC088D425405CC7FA3EFBD754C04F5F81C03BE0CEBE2546D4C09E29B9C072328C4029B0B23F49451740A33203C0F539B43FB341A1BFE74DE23F5180A63F3BDF223FAD16CA3C3D1A08C0A835BBBFC23511BF3DD3F83F85F161BF258C943F39AF8140BB372BC0DC2944BCA8848FBFFC090CC02C925E40ECC381C045D9E23E5EB98E404B23B2BFEA8624BFCD63A54037B2AE40843AAFC012F4E7BF4FDC65BFD00FDB3F0BD088C0905A46C00325C6C099B3F13F15FA3D404513E0BF6CD9C8BD641DB2404A33D93EEA5F023FE48FAF4085E98CBF63A512BFCA8B8AC0C8A46840260CBCBF94A60AC015B23EC08DD4943FC9B53F4002DC5F4073284F40C5A46B409476D4BF4A6136BF4395EF3F664CF13FF97B94408EF148C0D9241BC0E60939C065B571BDBCFD783EFA4D9140F578EC3D394D583F078C4AC0362505C08997D1C01984004014C2B7C00F86D7BFA42A0ABFC84B51C047E1DABF0D5A3CC0AFE246406C05F63F30B40BC0A8DE69C0F6A842BEE2EB9CC0CD3C96C09A93DF3F8BC815BE2A165440F2F7A040F8697EC03A150EC00B97A5BF3D1DFBBF8F5B8EC0B3F088BE939836BD8C29EE3F43E371C05BE5EE3F569FA9402D4D01BF9C8A2ABF0D6E10C03251A33F0AE037C0BA1B7BBFCC560FC0CAB27FBD443125403D932A40DFC28FC0D6E05F4056C78540085332C064F8DB3F7F2A4340CCCB82C089DF03BF"> : tensor<3x5x40xf32>
    return %cst : tensor<3x5x40xf32>
  }
}
