// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>)
    %1 = call @expected() : () -> tensor<2x3x3x2xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 3]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>) -> tensor<2x3x3x2xf32>
    stablehlo.custom_call @check.expect_almost_eq(%2, %1) {has_side_effect = true} : (tensor<2x3x3x2xf32>, tensor<2x3x3x2xf32>) -> ()
    return %2 : tensor<2x3x3x2xf32>
  }
  func.func private @inputs() -> (tensor<2x3x9x10xf32> {mhlo.layout_mode = "default"}, tensor<3x3x4x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x2CA910402A60FF3C2E11E2BF6849C6402CD57E402DB932C08100CD3E38AC6C3F43798AC0F0D58DBF116A42C0FC498FC07A33AF3F3C3583C0E95D0C3FC8259A3F0653BCBE5B397840E77585BFE0A94E403C456D40E35D25C004AE9BBE8F15E3BFF4C80CC07601D4BCF67026C0E69B0C4076E58DBFDEE6A03FF1D41BBF760F25C0A6A797BFE4545D3FCC7DECBF7D714DC0217396BF74A2DD406FF8DE3F6A8482BF1D2E0C3F4E9883BFDCDD50BF10B807C1B4E47B3FDF1735C05027D3C09B8739BFA9F7B2BFC95E8D3FBC558FC0A6E86F3F083950C0638D454006C2D03F475EAE40FF0027C0E24D21BF7D12E83FFE8513BF11AD624096430240D024BCBF359B2F402C368540D5E88CC0C547A13D3C0467BFEA620C3E0F58DF3F5C2DB1407C58C6BFD2B08B404054C23FFE77EDC089C4AABF5E4A183FF0CA2B402D6213BFFDF2AF400CA632BDCCD69E40F0CD05412A7A8D3DA18D7CC0AA7289C0591C5C4061BA23C076FDFEBFA646E5C0E4015A3E53768AC06298C8C02DA601BFD67007BF95DDEF3FFEC398C0D07380BF5EE6E8BEBB9E3E400A4A4740425787403D3295C0E405BDBF2840A5C050D3523F295304C182E403C024323DC093D3E2C050A8AB3FBE8E0F409D3745BFEFD103C0A73C41BF85A537C09793E740487CCE3F335F2740C0248940FA1369C0CD24673F1D7B9E3F82BF3340139D8BC03578FEBF0359ED3FCFB11F3FFE44E33F90AEA53F803E0A3FDA8477C025BDC1C0561E853FDAA894BE52D3084039A8BF401EC067BECFB026BF91ED7EC0EF3CC23F77451B40F07142BF4B6A78C06BBB3DC0909850401899EBBF2FB54AC0BBA3F8BEB493FDC02B34EE3F415F79408FE68640405384C0F247E7C0E6CCB4BF852CAF40A7214240F893884078B31BC035EA2BC06B7A843F9A619C3E240C4DBFFFA7343F2DAA5F3E7458CC39F616913FB5A1DDBFD8F41C3FD28C414059A6EC3FC180753F760CA43F250B75C09196A2BE3702B4BDA9FBA2403B4E5FC048CA3F405F194B40562683409220F2C031EFAF3F5710A93D43B42BC0038392C0998012409D1C943E04E843C0226A81408B0B2AC0E0F520406B7B093F8EFDB74083947B40BE2E0DC02E051FBF4E7ABBBE3539A1BF2CB6603FF949ACC009622D3FD369183FE3C0A6BF209F4EBD80A9BABF9986ABBFFBA54BC07CBFCB400C2E33C0B57C0B40809FEBBFE1083DC0140A943F940A94C05BA983C0DEB40240E2B2B93FC3C4E53E1B1736C04CF418C04350A1C056D69C3E6B1002BE5043BFC00380A43EDF11C2BFBCB94E401CC954C00C407E4004A10DC0B0DDB940185756408AF5384008808FC0D994B4C02C8754BE312C6C40E922E13EC6856BC00CEC7F3FB024A63FC6D7583FA08D98C075364A4035B862C0AC9B1FC05167A3BF3A06FB3F80E370BF6D3DA6BF00A448BE90B62DC05571203F76E26BC068F9D9BF7C4A8BBFAFD8383FB12D2740112EAE3FCF23033F5E5E3840A3A1A8C04DA5763F8D23BB3FCC09FBBF4F3C1640E54ECD3F5973943FC4976CC021C440400769EABF354CC93F59786A3FCC129DBE39E1E6BFDE2AB8C0F15224C0302F96C0E72147C0DB3CB6401442723F09D11B401207E53F2F8CCD3FE916FBBF5121BABF1CDFF63EB4F74640488FFF3FD5CD23BE96BC853FCDB49BC035663040551F064035C0D0400552AF3F39112CC0554A56C0AF4976BFFD6810BE898190BF3375C13E6116DCBF573A14C09EC24640D77A1ABF03BF8640EB351F40F1B5CBBF989281C04317E83DDB2AB63FDB728B3F8916E8BF6B3A3640D20BFE3FDE49424013E94F40D15B32C0DC467140C6CE54C0D338D03E6D5B623E58AFEA3D90F5AB40974587BF210F3B3E321F6E4047F392BEF2758340C9402940F3DA77409F3B483FBAFDF2C04E16713E43A978C05EAAB63F5A8C3FBFF814F2BFEB1D80C0AF7312402C231C3FBBA239BFD5B887C07186BCBF88260B4098E75540D4441940442349BFBA76E5BF2FC5A7C0FFA9A63F6B95393F632FC73E994A3E40353169BEA470B5C049E1473EA6C730C0A1DA9D4024B4913F18905F408EC826BF132846C0B77779409347AB40A13C53C0617467402B1AD93FEDEE1F3F1C3971C032DB294029100DC0032856C0D64AC13E1D3A38C04962BCBFC590D4BF7150A6BF7303593E69CF933FC7356E3EAE0961407896A6405D6947BF5A33BC3FC98596BFDA3137C0993810BF2D9E94BE406A0C402F90A4BECCC48EC0E3E32C3E290311C027B65C401487E1C07E362C3FDAD4C6C01B238D3F4CA38E4006860D40F88DCE3EC0551DBE67E4D63F243DFABEC98CAF401DFA8AC0417690C06F85293F7B289A4090827E3E89E02C401CE04EBE1D5B1141476136403A5E973EEEE0A1C01C3D71C096F123C0DFEC0A4077121F40423F27C0D2E9A14001D292C0174577C0CF1399C0F855083EDD1232BF34A88EBF771413C0FEA9D640002DEABF330913C17DE87A3FE5A544BFA956A7BFAA0D0F3F470536C0D78323BFED1750C07B2763C07122ED3EC0956FBE61AAD340FC551940BE8C073E11F19C3F2835AEC0C632F2BF765715C06B50593F287484C09CEFC5C07AC970BFA7CE28BFB56B07402683F73F4D6E814047142CC086ED393F18FED1400369353E27284E4008884B40987B8DBF5C770EBF80C786C040B6AF40D30815C0402508C0DF8212417B2F46BDA513C740FA613140948EABBFD28904C0EB76563F8AECE04096D3C33F3539E33F94208EC0136275C0E56D6F3EF132A840F0CD7A3E669521C068FA884071DC763EDCFF8EC02F9A9D403C9B3FC0C3B6B740DF051340A73C733EDB56B23EF6EF4C40C7F1A93F944FC1BF7D025E3F4BFD31C070910D406E2B8440A83B42400F96FEBD053C933FF6B44A4049DB793F9FCAB3BF2F4E233EE691723F87D2B4BFD0DDB94010194840E32D5DC0703194C0DF9B0C3ED331BB40AB1E3D40BE6300C031425A404B9B85C07E0B6BBF86DB92BE930110C05C591F40209F30BDB0E1BEBF60BE4C3FA376E63E98444CC00F42083EDB8DDABEDE4BEB3F09DF9C4036B107406D65E2BF9DB089C0"> : tensor<2x3x9x10xf32>
    %cst_0 = stablehlo.constant dense<"0x936416BF097C0A3F7542D2BEF62E433FB6FD18BFC785364091DDC7BF2D75A5C0839280BFB7CE0640EFE933C098A7F43E8FBE2C40900318C030D5A83FC06BA2BF153C95BFE225EB4020631D402441A63FEDCAF5BF5F87CD3F475DC7BF33E521BE2F0D0AC09D7F9EBFB2A52BC022EDC1C02BBD81BF1B2A52C06602F13E0DCF04BF7BF7B53FA78A8D3FA03215BF4349A1BE1A3CB0C0EBAB42C06D1E6A3E3D38074135D8C53F30CF054052070540D44D09C05D39A14094D3E4BFE8D37A402569654062E4034001F34C405CA2F43FFC8E7AC0BA938AC0F83FAEBE3D2684BFFED8AE40E576384056BD784015FFC73D84209C3F21AC7A409B670340E74528C0F2FE8F3F4F322A3EBBCCCEC04E1C8FC0B0DC9F40F80AA93F9972A03F13BE0CC0B6C78BBFF34D29BFB4E41A40682904BFF175F33F77885140579C90C08AFAAC3FA473B33EF0AAFC3FCC2AAC3FDCB45BBF386B5140752721C012DFA0BFA101E8BFB8B176C0EF3EA040E5C41E40B155A6BE9C9749C0FED6BFC09269C53D03DC16BF978E81C02E768ABFFF70D0BF403991C0007A3CBF333637C004DED1BF7C9E33405B0326C0AA766BC0F8D1953F0C0A0CBE0C671FC01217FFBF96D39FC06E54814096369FC0976A993FD5A0B5BFD9B8C63F198A2C403B0B5B3F00528EC07C88AF40AD85F0C0564B30408CA094BF802E08C09B84CCBF32E7AE3FCF7D1FBF34731DBDEF5B4C40DBE0A13E337C084030F55540DA7B76BF51AE64BFE1B793400254A8408A08874012FE15C05777A3BF787EE13F229FA4BF8FA0F4BF4F8D97BFAE0CF73F7BC139C0C87C0C41BA1784C0FB595DBEB96FA33FCF6A3EBF20C084BFDE0E294000C496C0BEA8A13F730A00417A20BABF00918740C49102407B30254008C4CF40BF578C3F0D2F04C0544135BC71F163C059630C403541BB3F0B63ADBFD73144BF9EBB5F3FBA61793FAD23F7BFA35887C06165363E1D715C3F6BC670BF40797040BF01B4404DDBE7BFF9836ABF81E51E40A50E62C0"> : tensor<3x3x4x5xf32>
    return %cst, %cst_0 : tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>
  }
  func.func private @expected() -> (tensor<2x3x3x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[[-18.2068977, 75.7893524], [3.128350e+01, 125.058983], [-1.73573875, -89.313858]], [[-29.6063423, 11.3755112], [59.1143723, 28.0051346], [-141.138977, 31.8165207]], [[-55.3804398, 76.877327], [-7.82424163, -74.3266678], [46.2979965, 83.5255432]]], [[[9.73925209, 8.08479309], [-85.126091, 47.5552979], [65.789856, -82.2686767]], [[64.2616272, 43.2840462], [2.391430e+00, -130.651321], [-120.757233, -23.7799988]], [[-17.5342159, -2.761110e+01], [68.806137, 101.990311], [119.880096, -64.2099915]]]]> : tensor<2x3x3x2xf32>
    return %cst : tensor<2x3x3x2xf32>
  }
}
