// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x0x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3x9x10xf32>, tensor<3x3x10x5xf32>)
    %1 = call @expected() : () -> tensor<2x3x0x6xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x9x10xf32>, tensor<3x3x10x5xf32>) -> tensor<2x3x0x6xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3x0x6xf32>, tensor<2x3x0x6xf32>) -> ()
    return %2 : tensor<2x3x0x6xf32>
  }
  func.func private @inputs() -> (tensor<2x3x9x10xf32> {mhlo.layout_mode = "default"}, tensor<3x3x10x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x31BD63BF504024BFFBB7513E32A6BDBFF46B79BF417AB33E6D482B407B84863F75F109C0E7D231BD171C443FC32A8C40C79CD740AC55CBBF29FC4F3EE42790C0726A833F5E26953F5D4810C059118540F124D44028B2A3C0F06EAC3F626DB8C0E1D5073FCE3F83BF95BF95BF68B065C0E20290BF95A46640F50280BF38333E4088F68C40AB2372C0272A26C0CBB938C0CF690DC0B4A889C09D819ABE3F8F523F040FB73EBC8D0F40D9D30DC0289291C0D32D20C033DF7F3FD0A03940F317E0BFA36C893F61484ABF1EFFDBBF741B923FE97676C0D92286BF6B78003F23BA9BC0FF368CBFEEC4073F0E5F60BF921BB8BF62D392405ACA48408AB670400192FD3FE3DF00C13B211FC066F65DC0D3B48A40BFE41B403CC9DDBEE154714092C6193F3263BC3F679F20404AB25F40FEBF8BBF5E021EC107DA4840E50E324095EBD9BF582218C09A36833EAAF48EBEC7DC943FA216D5BEF0E704402C648D3F09A8FFBF0212A13F739CCF403635F2BDBC845CC086283440A75D64C06E76A03FA2460EC0BE42343FDB768AC0F08C49C0BE461B405B932E40623E9C402ED59BC018F8A7C0D3472D3FE3D9633FA6DB3AC0A31153C05625C8BBAFCE2F3F876C7E3E7E7B383F4AF202C0B55D7C4026582AC021C570400001A7C0D59F04BE8F9BFD3FBB0F104013BA09BF942AF7C0E0D68BBF653A1C409D2C31BF02E56BC0FBC838BC0B7B91C050140B4095D6A43F34BD83BFA23D4D404ADDEBBF971017C0A59136C0C222963F392BD2BE9F0293BFD99BD43E7CB2243E0B18F63F455F8340AD260040ACF83B3F0A441E4065625CC0DD7A11BF4AE9A1C0FD205DC0B8489D3F02E28AC03BE173C00B191FC02E4040BF88A29E40C685A23FF81DFEBF764A863EC41D3E40DB768CC0C436943FF506143F6A8594BF5BB999C016DD8840648458BFC015B6C07C3E5DBB551CD3C0A1BFBC40C009EF3ECAF713405F5374C02EA82E40101AF03E5C300BC0CAFAD0BFAED4153F84AD753FE7AA2440B38CED3E46284140D817A1401DB630BFFF59E740CBE68F4025E1593ECAF11FC0B7920DC198D783BF21F280C0635D02C0CACDCA3F3AC9DBC09711A1C04D5F99409D22BDBF9449573F7FB23EC04C7C1EBF5829C7BF79DA89C098BDA1BEE54D5D3E37EE9A4092878B3F5E4F6440664FA840CC174D4059375DC0151B8FC032D0AF3FD3ADB43F96B265BC3A03514053B763C059161BC047410D3FAC781E4081C49DC05E5FFABFBCAB3A4034F7213F7F8F113F7C828D3F14710EC0F264824032E589407A04B240A012AEC0769E4CBF39696A4017D91AC02AFACC3F60E239C0E8AA833F17D9DBC08E8707C02FC824BF67C924C0A090BEBEC6798FBF976AC1BE0050B340EA5F04C0D72F8040C05EBCBF188BE0BF1FB87FBF782F2240B419AB40F9B31240908987BFEA8A9040BB137240F1D24BC02C7A993D384F8240945E98BDA0597AC0D1125AC04E6F103F2867013F50B2D0BF626E9ABFF29D13BF18CFB43F5573B63FF02B554085A97EBFCB26A5BE6CC38EC09D30D0BE6C3CE2BF6743DDBFBA4C60406F85E3C01449B2BE8BE43240B7B30C400E9911BFAB26ADC05DF611C04DF8873F750AD13F3AF35FBF8F7F5AC00229873FC6619CC0313050C097239C40ED415E3F61E088403E6420C04C29CFBD8233A5BFD5D6BC3FEBA461BFC375EC3F0F25BCBE46F1E0BF6F1C02400870D23F705AB23F4E6A453FEE39714017C41B41C10C763F7CC5FCC02DFFD7BFC305F1BFEA10DFBE1245D13F833965BF90F088C04AE587C0E181D53F7BDA3BC0601EB3BF6CFAD83F6F497A4031419D40ED15BBC03F0DC83EFF303DBFA75D4740DE950ABF8277773F56F4BCC032D94EC051E7F6400DCDD63FCE6D27C0025F283F5652A63EBBDBA03E10B1F340625E4F400C2286BF308A903F7548F0BE554BE63ECF22D5BE24D304415C8B4B4002CBC4BF139451BFAE6E44C0B3D42CC03A2C11405000913F80F615BF8AB40BC0902B074040D32E4095F99E40196F8FC0617D684023B646C00485EA4032D5D1C0D82271C03E836DBDBAF15640BE9072BECDA9573F17F87740BA52BD3F28F567C0489C8C4086B27F3FD018D0C02D21434029B250C0DE577840F06E66BF991799BF0F27F93FDB2248C0200D75C0CA7CEBBFD111C63F9BE269C04CFD813D1353B2BFC95DE7BFBD2188C08E08423E694B743EE34279400ED469407B5CBF3DF2A00F40EE8726C0AF3A9F3F0E1AB040BA0424C0758528C0E7AD773FDECC9740B9E19E3FA3C1CB3FF949E03FF8147A3FA803B53F7434BFBFC5216DBFD5F62DBFDE47C23F235F6D3F244D3F401E712CC0C7D6B6C088538CC0E4EE36C0376845C0BCB907C07796FE3EC32A0840A4F7EEC05507F4BF87E755C055C718C0D2BB7D3F7BDE3CC0EA5F0C403BA218404F158E3F06825DC07C2B07C0397AE0BE730DCA3FF67351C0DCE96040D6FFC7BE77FAEC40E30C80C0606427C091AB65BEFE83D13FCA7F30C0BD74B13FCDE57CBF5F089ABFF4110ABF24D6ACBE70B9FC3F9F5F0AC0EAA2563FB7338940CC501240851EEB3F76061140F01C043F4A96514049BED440A0075D4037F945BFDC46664037DFB1BFD6E33DC0450D57BFB2A80CBD47AC28403C5461C06688D940725F17C0EEBC41C06DE235C04AEE493F1637284032BC5A3F6916F3BF06687F3F1C513AC0D8876E3E9AA769BF63CB05C10093E5BFE3F323C0C5503140B3BCB7BE53D14140F61DF3BF7B1835C0F370184064A30CC0E719104010ED9740CB15783F3258D7BC4DF682400697283FFA893D40AFECB4404D3B2AC0D290C23F7B5D8AC0C5D804406BE00BC07D576E40963E0ABFDA4033404F589BC07CA794403060643F691DC04081948DC0C4197FC0B59A2A40CEA823BFCC844540B5DA8E40D78BED3F8E0BD53F8AA299C0E485B53F4A7903C0EF22BFBF1CAB4CC0DE54E84077F7E4BFC0A61240B6831D4056A3AE405B21BE3FAA9AEB3FE0FB9B402E5505BF45E9FDBFCC13213FF27B074040B4C1BE9307F2BF5BCDB23F2C2B943F3B692DBECBEABB3DE07DAA405B738AC0"> : tensor<2x3x9x10xf32>
    %cst_0 = stablehlo.constant dense<"0x83CF1ABFA98F5B3F52F88DC0921D16407C8AAD3FFC811840148272402953BD3FA7802440C71025407F9AA74085BF5FBFDC60014049A094C0BC8FB3BED69194C0DDA6D43E23E6CBC04DA15F40FBBFC0C03FC1773ED9ABB6BEC3E09F3FA3329640C37F7EC051A91B404CBFCEBF86E63C40CD54CEC0F07D6040D88697BF3680BCC046A27BC02DCA0D40418855BFBD07E7C0107A0540F14BEB3F0991D03FC0D81DC0FBD98E3E6F5122BE377C8C3F342D93C09D4299BFB0BB5F401267FD3FDB85803F2E8451404E158D3F1926483F8E2D683F78F1D83EEE254E40724BD7BF5D854DC014339C40B64B2440D7E9BFBFDF9F843F80BC5BC050E6EDBFE1587CBF1750283FB6C6583FD777BC40E6ECDE3E3C4C96BF7C26BDBF290581C047E3C8BF420DD2BFEA578DC0A55B83C060BB6B40ABCF943FC30554C027070740038984C0BCD14BC0122D77C023717AC023B7FEBF263B98C03C8EA5BF69333F3F9591D93DCDCF2940DB1056C0711282C0EFFF4640B4046D3F3238CF40DD4285C0B764B23F442661C074B82A40047A464044B8F1C0254633C07BD74940954F0340B25DA940E36065406DC829BFC9059E3F092A53408E3B31C03460C2BFE68B5B3FE94D62BF925EE3BFF7A661C0C324F43F49B1FCBBCADB0F40C43813407CF6BEBF3C57ACC0AE0024C0D944523EC7F5454021FA8D40B4F1F9BFFB9D27C031E61FC00165034055CAC1C03DCBC5C0C13012BF7A7E64BF1695553C55ACCB3FF84971400636713DA62921BE033EBBBFFCDF0FC0D083543F1A6CDAC011F1E63F25430AC06148C33F879D2240C57D90C09AD377BF304BCFC0249E81C0886E8FBF1B9AC83FDD720A4017C959C026381AC0F6C24ABFC479DA3F8B1BAF3F2BD40B4097EF4E40D90C9440FAFE7740957010BF6F715B3F8C79F8C0497FC63E4A9284BF35FE9DC03FC711C0E00E5FBF157A3EC0F7537C3FFB6EA0BF56DA0B4100EB0D3F4FC09DBF5A04E54096CA2AC032448DC04B2123C0732DEA3E48B11ABE4E724240793B01BF2DF62D404FA179409FC3CB3FE0AA3EBFF9CAF2BFF63B03C0488ECF3F78459BC007942F3E26D9EF3F3E4ADDC02F6151404322DDBF6C3F55C06D3902C06F7E043F7CE3053F5E3B1940A652FE406C7568C08713AA40F0EEDC3FE97AFC3FABA206C090A6ABBF5FAEEEBFFC6AC74082FF5C4025989BC09C8068404D4F89C0E98740C0C1F60140DAB4863F9D1EFABF6D7B583F54C65EC06366A93E6F8DBFBD4141A2BF731EB7BFD7358340E6A25B3F02680BC0A9CDCC3F1D56203F2B62843FFC170140AB4C1340AC213E408E3B27C00E0F4EC012ABBF3FB2550A40073B3840D7758F3F620F5C404CC64D40447CEFBF5E624F40727B39C09D7066BF7D797140DDC428C0AA5914400FDBA5BF18E340406A08683E4A31E5C0BF1141C0AC8FA23F7A74DDBFDA474D3D0C3811C0CEB859C07FB6BCBF68EC31C0ED0D0D400F0F44BF141AC53FB928B5C0E65FD24056F105C03A39943F8284AE409829DFBF9B3245407D38E3BF2D33FC3F01D8AB40C0768BBFDCD30BC0FFEECC3E799E8C3ED6432F40360336C0E7A394C01BC600BDEEF1E5BFBB025AC0A998503F53DC7CC0ABD6E63F22546E3F1A45EAC0860EF4BFD00B9040DB75DDBFDF714E3FC46CD6BFCD1308C0FFE227BF38F42ABFBF32A03FE4814B40E3E2A6C037469DBF200BA3BFD27272BF00533B4052C1234022D899C04158B3C058E480BF03160C405EF164C0253868BFDE3EC7C049862CBF5908FAC0FC0FDD40BDA2E6BF5ED986BDBDB75C3E6E78FABF16C2DEBF787D243F24AF21C02195C8BF49A858BEBE5996C00963FBBF2E6AA53EE888C5C0E66BEE3DE6B7963FD1928F40A87104BF61EFD1C0119459BF46730FBFD1420B3ED84174BF9BE4193F19937BC04EBBD53EEE1AA140780843C01FE6F23C535700C0A46B3F409519993FE6811E409F2AF93FD63285BF08D4323F53FC5B3DBDC20D40A4D67C3F609984BF610BE83F5BD2383F9ABACD3F73340440CD4121C09DE6483FA4B71040CB5D4440C5F702BF216085BF0D1521BF43BF6EC08113613F4A3517C006D57CBF83871FBD0F5663BF2AAE2FC08FF151C03434173FEE81173F5F6390406A62F53E83DC0D406C48303FDA8482C006E17E3F5B7A7DBF26AADE3F905E36C0A6C3933FC8FD1EC03F6F4D4046878CBF094EED3F318A1A406F8757BFEA18A8C0D64757BE03684C3FA27F1FBF7D6011BF7B777FBF0797BB3F23497E406012AEC0754907BFBC5DB940A68C45C0B365B23F06ECBB3D38613E3F089E08C0F98309409840A4BD8C4275C07446B9C0176F79C06F614FBFE21CDCBFA0115040416745C08E8822C09DC50EBF884194BF82A1A840F0966E3FFC661A3F260073C045353140913FE93FB8D45240F5FCA7C03E465640567A2CC09F1B73C004CC39BF537180C094BADA3FAEBAEBC00C2D25C04024D9BE601996C0D8398B3F08F0903F2F94C74039C46F40D4677840022D2B405FA158C07AAFD5BFCA3636406BAFEB3F2E0743C0BEBB46C0E5A32CBFF8F30DC07B38513F"> : tensor<3x3x10x5xf32>
    return %cst, %cst_0 : tensor<2x3x9x10xf32>, tensor<3x3x10x5xf32>
  }
  func.func private @expected() -> (tensor<2x3x0x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<> : tensor<2x3x0x6xf32>
    return %cst : tensor<2x3x0x6xf32>
  }
}
