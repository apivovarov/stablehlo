// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf64>
    %1 = call @expected() : () -> tensor<20x20xf64>
    %2 = chlo.digamma %0 : tensor<20x20xf64> -> tensor<20x20xf64>
    stablehlo.custom_call @check.expect_almost_eq(%2, %1) {has_side_effect = true} : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    return %2 : tensor<20x20xf64>
  }
  func.func private @inputs() -> (tensor<20x20xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x51C44FA61AD215C0F8BE0423B402EC3F17C7AE8AC3E6D53FACCEE87C3149094000DBEE8CE9F5104079BDC987D7CBF5BF10512652D244CFBFB01E89DA6CEB1D40E6736FAB106CBCBFF2410055D791B53FC96E396F362A0040F9F07C43C9EE0540549A3C48823B06407DE9D21A881DF73FF6DCFAAD1732E13FC282024985E72040EA5710A0C1FAFA3F74C63DAF6C1FF53F6DE8CF202B411140AA515170DD150FC0AA0F0CAA5856D23F5256257537031CC0064D6D4D19AE17C039D45C1BCEB3F4BF280096C90FB607404DE0C913071EF83F2B7DA1301FE40140ACC7D8C5871AF83F8A6E98E5A7CFDE3F705ACC26747806405D139F94759B11C05DDCCC7B259DFB3F9FC56AE14787F93FE067F5D4A9B502C092E1D0986D6CD3BFEC76A2995D8B0040CA458EE042D009C0DD403375FFDAF13F506555F314EAD43F9ACB05C17F131540286308BB0C44B2BF740B0CA1D4781140F90534844A26FD3F86510ED2A607F1BFE8C9B1339F1612401E3628542E73FFBF36F95C571BC4EA3F9B8E4648EB290D4030DC6F2CAC4EEDBF023DCEFE6C59E4BFD0147A6080B806C017DF7362C251EFBFF48FC54081F6D0BF16E6DD4D59EECEBFC28BAC671E6CF83F6555E12195B0F43FD452B24538D8F73F8A014C5A9DEA08407EC3FCEFB426EABFA8E708DA7AEBFABF62CBB45A482C0A40C8C22D0F708E0840B318CD3626390F40E0BBACD95F13FD3F00BFEA79E3B8E3BFCA0D6A7B99AD04C06B05E69B837E1A409C34C90AA5A403407DA362038B5CFFBF240B8323AEBE1840588E1DFF1564FDBF23C224E5CB43FFBF9CDC60E496E60740F1824CF3508E0B40C23EC543571D09409D1EBA5437A2F13F763648541D900340AFFEAB1D9A20034073DB9918EF66EEBFDAE51E73538419401C199668DB8BF0BFB9A332AE97A21040A9CA2C220A46F43F7187763992BE10C0EFF0249A8D27D4BFF6A32C41CB99FB3FCED824A265B91DC0203C46D2CAF91840FE355179A236FEBF3063753B97C610C083E03AFF9540D9BF08F5B5D8E97403C01A7851564433EC3F6E2687105F3D044074C9C59701DA07405A199EA4A464F73F6ACA730E3F8719408094DC475ADE074040B5359EA9B1FF3F643E1FE1B64A1C40B2793F9FEF7FE2BFC9C1615DC96C0940CD651D2AD162164090C6217FCC9E054072F433AE5ADDF3BFF7B00221BD6AD0BF98EAAE9B3AB602C0780FF4BF4589FDBF314FDF25C2F9F3BF78BE693F22A89ABF6DDA5795A56700C07B3349954BCC02C022DF23EE3D3916C035D7FC9FDE31E43F029878F53E020540DAAFF80974A212C0183C7F4FD91103401EEA900F2B0410403492697C2068EEBF5478107CDAC6EA3FF4E283916F7FFEBFD82030D53971084075E5FB3AF20411406CDCE4930691094026022C700005D93FF2BA51E68E7CFFBF56CC3325FDAF94BFC08852B8F7450BC0ECF7A334CADE04C0C530E9F3DEB011C08CF281D7B6E7024032DFC396C4A8DBBFC1D8B48E116E07404A88EB5B43FE0AC0B3BF12B65CBAD33FB89DBBDA3660E53FC2D46AE9DBC7EC3FD5845BB66DE515C042E996BEED3BFABFB6D5A998AED8FABF8A7D1179C1F61A40C4392381B3F10340F1320058C7C118401B0FCA812FD705C0D0447AF0862CEF3F68FC6CFD5D1A1AC0F04C0CF9C50F06C0D0C77E7F18D6DE3F9D403DB23FE8F63F3A46FC9E97340440C56E02F0CD13ECBFA2364E655ED305404ED1AF1D9966E33FD8F24D882FE40F4069D14FFBA3BF0440B41BC37E8C9417C0B3DFA80EA564E8BFE114A0A087D7F6BFE2D9C08BCA0C00C098C312789AFC0CC03EBE441B3C6F08C0F42DE1B5FEA5F8BF4E7223060B70F0BFD4FF2E1E440C0E40BB3CEE1E0600E63F9E9D51108D98B9BF24C5E4222732D13F7E35347EAD64D13FAFA6CE82D779EC3FD7EDFA70D8D310C09BF8CA0511CEB2BF3AC0B6DB700F16405A47B3058EB301402F4A34ADB1F2CE3F40167F28788C14C0F4B1E859725ACEBFA29A8BC6007DE2BF61E9827ED68A0C40B8DB17EFADE112407AEFC44E946312C0DAC7F36C280608C01E423E6432F612C004CBF1958F3F0E4004E75DCD9AE20AC0E2A067BF7528F33F5E3424024404FDBFE431ADA31FB912405A0A6BB4A334F83F3AFB505CEC830CC0174A0A086D8A16C0D611B5581CACF0BF5A17C17FEDD203C0EAE7CDDB4863EBBF9CDA120F2CCFE7BF031DC19C2C8A07400A6BBC7036A7D53F4A1C92DD275DF7BF139F8981BF2EFB3F2635E1A10554D1BFCD060884620EE33FA049D1FA424403409B45441D4EC1EDBF766D824B2DEEFCBF226D881883BDE33FA8AD7BE1C54BF03FB4B9EF5123D90A40C616214333E4FABFD66B57684F75F83F0DE57451FC0109C0DE09962148500CC0B249693A474FC3BF8E9B9C13224A05C0F2F6121B5BC3DF3F3CAD9FD78C8DE9BFF474A0A42E0EBB3F30A7F5DF902CEABF322F8B1295E204C0884D104BFCAAEFBF5E880810BC4911C02AC40FE45E350B40FEF4E68F24051AC06D61FF97D04EC73FA653E595A99A0EC0A4E6118800800FC008FC2C70E62E1E40B4998CC0DFFDF83F3820E60A83E40040C3C1539B9EA5F4BF1899A521D48AFC3F71D2EE77ECB9004064EC181394440240FE9C09165ED202C096448DF5B5CDC53F4CEC012653CBDEBFA4BB8B9C37E90D40CEB7A0D4957EE9BF242E92632AC1F5BF2A315DC1D14602407884DDFCC8F1EA3F7A2B57D3CAC706C0AE48C5D77A6E19C03A1598122E7CFFBF215839A747C2D63F3FC9285215A8F5BFBFFE945F434E05C0C581E82225D608C032E5140791B41040E2CB5F268D72DEBF1834B9C3C473E7BF1C4B9F759127F4BF544AD0988E7BF93FDE3F26CE5EF215C093FA806711C003C03A5BFC126717AABF2F4C5940811311C0306D2A191772F43F9FF8D667AE9508C068A94B195CBAE23F97F77DC9EF2019C000F9ADE2BF3EE9BFAAA47383EC9CCA3FA03995E98E160040F2BAE69422AAE2BF005B457C8AE0F6BF9F0EBB1542D6F4BFE74D81BFE6E2E2BFBB5A7594E24BB83F3C31FA3D9A56FDBFDC37A3D45791ABBFFB3FCAB14AA900404877FCEACD89DE3F179428C495260EC0548932596361ECBF3FF8F1BADCB5B43FEC7DFEFB0E8B15401F05EC775A63144001DB81651161D33F9AD035AE58D5E93F18366831101CFF3F1ADF6854900309C0AA3B26EBC76DFF3FDAAF79CCB67D09C0F8894ED9DA0707C0F0DE09A7AB8B04C053B1DA3641E40EC0733615BEB412E3BFF143A1C05942F5BFBC319119DAD700C09C94AD2E296EF13F8E3E590BD47CE83FD2F040C72F6013C0BE7E409063DB164076AA5BE0E6A1D9BFB687DA4B37EF1EC0B8810A8A9F53F23F3C25FB0318A6ECBF423D4F63BB6017C03136A020E768EE3FF038ACD412EF054029F6F2CF54DBE5BF6CFB639AA34803403C4FEAC0C279E13F129DD26F20FFE83F7EEA2F072E6FF13FE79560DAA9EDF2BF56355123943ED43F2E436A179652E1BF24BB53E0C670F1BF0FE0C538364402C06EC3FE76B6AF1AC0600DF3B232D8E93FAE6258659CC9EBBF68E18D7036AD14401BFDEA0D65511240440CB5E52ACE15C024EF7263077C03C04A2C0D44CC3CCBBFDF0DCE0EC2E81540FB96CA6FB47E01C03C80D96C0791DABF60C807544B0F174098F80E52B8E1E13FE63F2BC449401E4006AFE6A4D26315C0183169450129DB3FB32C03294905D53FE0F70D8135FCE8BFE4C9C9C590E712C0A900B769258506403DEB86C7FF3903405A0FBE1014A40CC0CCBBECEE3A4E0E40E7C5784F822F0240B6BF81C56ADEFE3F388F0838907CF23FADCB1007633F0240FC5FD0AE62AB1840B808BB7AEDCDFABFBA61E2257E43F1BF15C51B04E12AF4BFEACF5A3BF8AA0240AC89FD2465D3DE3F1CB156061AE20EC0622F6F7C0BF104C035F6E2F6467CE7BF2405B9D28EC908C038B131B20D65FF3F2EACE26893AAE3BFAAC47EA9A88FFABF381E4BEC8F1EED3F62CE19146F8D713FC92DAFB76C320BC03E792442DCC20FC0521AE5457732FFBF2C126BB56BA819C0FEA71E99AF180B40E6F15A911DFF0F403475D477D44A00C0B492B390137DCF3FD00C800C53A31040342342CBAB08FB3F5265A8D0FD2910C0C8530908319BE13F0320210EEB781CC0B31240B07CA8B5BFBD5B7ECEBDF2CFBF6C1168F53D7817406E9CEF413A9B02C01FCAA52268581B405379D8D3379115C0E8E414CD80EA1040E0246CE3AD17FABFCCF7C6417E5E0A4074FE607C45160440723773028814C5BF1F37F052696810C01A4470EE690E05C070C919EB7753ECBFEC8DB85C9089E8BF247BBF6D9CEDF1BF093C602B039297BF83A5BD29C5A805C0F6BD0DCCF217014064C088E7CEE40840FC9BE069FD92D03FE4CBC486FB32FC3FF4CC75A7350701C0661DF1528C6901C0BDC21ED1369FFDBFDC1C9EC2EAE0EBBFF8B1C072646DF93FD942488AB0CFD43F68DF3C16B4D8F63F9FDA9E9594A0F23F703CA7B42B1B02400242B8AF00450F40A5FE879D4530F03FDA84CC65480CE53F9B27389E95B8F13F4BD85278CF0C02C0"> : tensor<20x20xf64>
    return %cst : tensor<20x20xf64>
  }
  func.func private @expected() -> (tensor<20x20xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x9C2222B48BD80140FC804E2816B5E9BF1868AD86365B08C0AC2F70B5D27FEF3F4AE830CB3027F53FD4888CAAC9AC00406666C3D7452D0840DC2892272A1AFF3FF5A5EBAA24762040228AF4604FA128C01067A37A4CE7DB3FDB674BDD4016EA3F363C828CF29BEA3F80176A683BE690BF4E7A860F2CA4FCBF2552B0FDC3970040081CBFEF9E49C93F68ED2C8631C2C2BF90B5129D7076F53FC68FB0F9818C1BC0C75BF546E1650DC06A2ABE829E057440D0DB4CFCA3C124C0FD4ECCA4CAC70740D748B5D2D011ED3FD04F81D8242EA63F4B876274EB13E23F801110FB29C6A53F5D3041BA2B7700C0BE46D3ABA204EB3F964D6F9B10BD04400024E5B67A4BCD3FD8D727219248BF3FBF8267ED7A560640D6D891B2228500407AA869B508CFDD3F86D8AC707ED41340F67B0E2BB3ABD9BF54FDD1B4379809C0A693B3C5FE05F93FCD59DCDAD4A02A40A82B9EF916B0F53F2C29C9F51443D33FEB029544D6922F40BD811BB2424FF63FB20285448E103CC08DB38816104AECBF80C977A0D266F23FC94EFB134F7A26C079321330015EF4BF045B653DD60812C06D319E97AA4447C06AD8E9AC8E2F054007FBAC5B5D960840B0F1EF49C12BAF3F5CC28D7707A9C6BF006BD0BE61079C3FF69C0FDE07F2EE3FB6ACE18C0D3C12C038308CA0C0D4F3BF7C3EC99B5665F03F70989AA37665EE3FABD8C9709BA6F33F0E62D9E8140CD33FCCB0923828FEF0BFAEC809065386D13F64DEEB341003FD3FE39DFD901ECBE53FF6BCD8862F0538C04677B27447D4FB3F06246D3CB7EB12C0FAD2680634B534C0BC989EB3355FED3F64F1B1CEB95AF13F1F8E0A98613EEF3FC098956578EDDABFE1C13E80DDA1E53F84C1CCAE27BEE43F5A6287B7B37833C075838350AB5CFC3F9A67DAD2839E3D4070BA9F44A2CDF43F40F24F7A9984CABFF5840B0D282C1940968610A26299FE3F54F682269C36CD3FCBFDF2A22E1A064007D679388CFDFB3FEBD3D48986D41EC016FA5FF8493318403E1182D4F136F03F8ADE7E2D7C2DFC3F5543441D3B54E9BF34F75232C6F7E63FE857DBF8354BED3F0078CFBF95D43B3F5949FB9EA65EFC3FA066DFCD1F52ED3F80FEDF949643DA3FF85527E08E24FE3F346CC15A8BC0E5BF1578B9DC8FB4EF3FEE9734D71515FA3F1F8EBCDB8A88E93F9E9EA4533D0C0F40BBA7EE42F45D0640C2B8792F614F0640F80205DB347A14C019CA0645B7F20D400086A28A8DE54240FDB3009503893440ABCECED8943F05409C273C2C52EAF33FB46F617727EAF6BFAE0E508FFE6BE83F306762BB6348B1BF8167ADF78E9FE43F492C2892C91DF43F19AA8074CB8733C07B18196C2A44ECBF0CA40C4520E922C02D0C46F66938EE3F6249E8012537F53F7AA4DA33EAE9EF3FD17E3D83A10705C0DB614CDF7A253EC0205F24659E71484032DAE9A3314D0240806B5E215477953FF37872F718FD0240876511B08D47E43F5B7C11FA86D3E43F28300DACA69DEC3F6CF80A486E580540244D4954633B0BC0F6F6398BD806F5BFED1E4A95DD31E8BF2ED26BC9B05C0040638832A7E68AE6BFC13120D0C5D8F2BFBC53B303B250FD3F80C716722A64E63FDC39DEDE73D6FB3FE47874211283F9BF978495D220DBE3BFE803E90DB921FB3F13A115C678E600C01E0E567CEF7200C0E08E0BFA5A129EBF508E004DC8E6E63FE12323CFC1A31DC04530AE4BF3E5E93F96222575B949F8BFB215142D3D09F43F6704D780EBEFE73FB32B7488704D1DC037774C0BE31809C03E5EE225B34DF63FE84DCB8E7920644090FA6A0C1D87C13F2B3946DCFE803340C5A33A4D1C86D43F7AE2E8A92C74424060B5503128F3F23F8CFE16166118F4BF200F5F13507E2240D0123D21C06A0FC0014C5E3FDD0C0FC0D39E3EF835C9E8BF8E6D5B7670BC16402308DE8D1FD129408E1F28C954D3F93F79F5918C1FA6E13F2A9958E1437E11C0FBFC83A2892021401A66739BAA4E0940D380CD8D5AA4E5BFE88EAB1A4201F23FD2DB00F96E13F73F72A2060BCB76E43FA455C1B03CDD74403CBB34C883CDF4BF2CB6F19A4E12F33FA2A9F25CD1970640F0995EEB28B6D2BFADD9123B621C0FC0EAE55530FEECF63F00EB32368CCCA83F841EBD8C7153E83F848154CA7E3FD93F8CDC32B5191C38401EA34BCDC003F53F045A61A42F8718C065056FB1F34306C06D4561E131CBEC3F32023A8277A808C02CA5DD301F3FF13F30C202096F95CA3FF9E7BA09CD6D0440688EE9A5B9E9F8BFAAA15965A607E53FC01963ACFC4A2BC04A9B5BDB4CC50DC0E6BAAB4293B0F7BF04CAB11A9582E1BF0E0923BAF7DEF03F2CF862039A72F3BFF0C86221131CB03F016B783E299F214064CDABE7CC27F03FCEA2B7D340161740048513EEA4CCE2BF6D3F9A4FD4B5FFBFF2008E6BA9FC0FC025C5925562C123C03C9D57171B5412C000A854F9F4334E3F63EED4DB3AFA57C0C2072C3700540C40C32EABAD6E1EF13F6DCC08DBCD5DFE3FB996988BF83617C084478145EF6D0DC070631D22969A2CC02304142D9640FF3F08C7B5FCC1C7B73F003E9A1E7882DF3F68F3CE1EE2320840D4813A89B379D13FCE0A3A53DAB3DE3F472E1919CEE9E23FA4BD49F27EF60440976F3F43A5CA18C0D5AAB93ADC39CA3F06A3E395BCDDF23F1E382B0CB6960FC0BA79001527EA0040F23590B7B5EEE23FFD5640C474E8EBBFEFD8E1EF995A13C0292A4FA8DF700B40861E9E8E850E3EC01A85A7D4235C07C06AD55F064A7C01401EA83E7E6D9DE3BF2C872EBB44012540342A37621EE1F43F99F403918938D03FFEBEDDF7D0B004C06AA169E36B460C402891AB5BC3A6BE3FAE72FAA714C1FE3F507E7EA0196EF63F53DF33EA80F532402664DCF8CE6611407C647AC5ECE8C8BFA8959B8C7E6F2D404AB558699F86F9BF10912B5FD8E91140773E8DC161F50DC04B3B893ADE5A14C054BEDD2AE382DB3F7CF8722B3158E7BFB8749CA3B9F4F53FA12B507411CB064045DE2165F682E9BFB1FC6B266AEF25C0A9CA9DA02A6512C07AD4EBC629E73140AC641C027862DE3FAED205BD7EA500C0FCFCCFF78DAB00C0343596EC9D3C20C004D400F2F49F29C0729F10C2B768F93F26357A541670F83F155F9D398FBF0BC0B1889B07EC58EEBF2E4F4E4064B8D83FE716318937852140B474D52B6991D93F30CA259F5D2E1840507DA2AAB2841AC0F4EE4A3E5847DC3F5F2B7DBAD81F15C02EED96D0515CEBBF598920BF28ED0340912F443C5B34244068500F719319DCBF9BA696FB82C3F0BF5D7B8CE431D510C0951C15A55972FA3FC0A1A6963B8AEE3FB88078849431E7BF049A549EA212D7BF7B6283839DB221C0C27AC5083C3E10C02C1EE3D2A82FE5BF9CCAA617C216EA3FF95221AD1474FDBF8E6918A6A010E53F6EB0A856C309FCBF600EB1AA9325F0BF2E900658A713DCBFB75ADCB3A88715400A7BFD24627F0AC0F73B336AB37DD5BF1902F3FFB19326404315844C3A960C4070419E958C79B73F77E42E367552EEBFAD970679044C1BC07C88BE909CAFF83FC95A323A0289F63F74B500A0A1260240507E3F29969FFB3F27E982420BA30D40384C8B9376B4F93F255E208435DD1640CDCE0D5DC6FEE93FA4036C6FDB99FA3F10A89B5C0131FBBF90250B056E4AFF3F068C1000E4310B40A41C749D012503C0B0BBE0C9C97409C06F44E3FA295E0CC0A1FA16A672C2F0BF098651D9451AEB3FDA9B05A590F2E43F8EEE4C346139E33F9D80621B2C1BF33F2A82429E8CBBE23F2E1E53CBCF12D83FECE854196536D6BF6025CF5F6EDEE23F78147DE6B7C6FB3FFF6F30E0BA4AF2BF1DDDDEF42DBF2940B04ADAEB9A280C409BAEA47C06C7E33F7F0661FCB57400C04AA0D615C0E414C030DB98A11B8AB3BFC1AE7A3F33D504C0BBBEBA18C13C2640309ACFA45D7AD93F496B3A076CB3F0BF12CF41027560EEBF7A3F0ABC768CE7BF62B81CB6AE3D6DC0CEDD37EB9E1B03409ECE2C5A3BE63FC0DE3A0ECE45DC32C085B0EA0869650640B64E8D7BCC0AF13FDD11954C8E18F43FF0C2F84A7E2F3C40861C3BD4583011C00A90DFD86DCEF43F14CC3DD5BDA2C93F74BB18BE18C43940BC9DE5AE25C3FBBFA55486562237244065DC5C3D4C302640A8F3C2DD8A5F0740EF21122798E8FA3F90D6562998AA074099E58BE7AD8EFD3F6F3C0410BC150740CCA6EFB40A1BF53FE7989BA3454EE3BFC65DA3FB0989F03F592E893CD4ABE63F90985A6E09BE1440AEF77612FDFE2540D877E114E968CEBF6A22C8FD95EE1FC040F062D566DA09C07C20CF6C76CA20402EE904A2116A4540EFBC98BE88A0F3BFCB74D448563CE03FB07D87013FE9EE3F60D2BFC1115010C0D8CB24FFB371D03FE70C6BEBC1A6204034C25ACA293F1840425034BD787A15C044D4FD1EB7FF1BC0B0E805D9C1E2BD3F2A42A5B8E7BA09C020E93E58F8F8A0BFDCBEA30BDA76D5BFDCD178A4A78EE23F37DB42F284ADF33FCDBC6C8318DBE1BFD00768739D88F5BFE646CC3F076EDABF9D79FBC0AD281040"> : tensor<20x20xf64>
    return %cst : tensor<20x20xf64>
  }
}
