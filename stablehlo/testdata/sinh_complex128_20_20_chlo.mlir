// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xcomplex<f64>>
    %1 = call @expected() : () -> tensor<20x20xcomplex<f64>>
    %2 = chlo.sinh %0 : tensor<20x20xcomplex<f64>> -> tensor<20x20xcomplex<f64>>
    stablehlo.custom_call @check.expect_almost_eq(%2, %1) {has_side_effect = true} : (tensor<20x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> ()
    return %2 : tensor<20x20xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<20x20xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xA8BBB789B54F12C06B992CCF2266EF3FAA467FABA4EDFD3F92B97C61E863F6BF7241EE6CFB71F4BFCC88927C5F05014026D01CDE3A34D5BFFDF1B524072CE33FF3D700D4CE76EABF92E59E4D2C8CF8BF4F966537151303C03800301EED6A1740CFBED809A5F21B40E656E003F09B04C0C0693A30CAEA0140B0B85AE8C9ACD63FB6A9D9F3A9E1E63F6987BC38C1D510C0043659B1E89909C01C1DE5539D4EF4BF94D633C83557E23F6CE85679B83BFFBF606F0DF3DC7D0A4014D7D3C1EB091640BF5523EF90180340852C2659F75A11406CC86847EF3ED1BF3ECAD4C72A02F93F41B80FA778A9EEBF2D8EDF9B148505403CB939DEB97F0DC008B32ED8DB660AC0647D0B089A340F4084A2923C9947D43FAB7857E7A9B611C04A76F0A0B447F03F7FDE05DB8A65E03F9E9D7DAE4C44FBBFAFA7620203700740B69E27A4F84AF6BF3E47D388CDC4FFBF5A9476985A5EDEBFA210E5BD66EB11400F2004CEEBA00A402703C69F49151040C9E4F08F4AC4C53F49CEB868FC6E104031CA7D42D43F08C0D2A177CE87B312C0D0BCA9CCB4BB12C0FE02DD88D9A0014068584FAD99AAF13F87BD10816AA3E83F1D504AA5A3E6D2BFAC2CC21077FD03407C4F0913A5CBCCBF9A163D3BFB220B40482ED78CA32CCB3F6CBFAF73E2430CC07553E45821BFF13FA5E006A9965412401CC157C55BEE12C074D04BF3E2FC09401D3A1B818B541540D21879DB2A4F194078E7BF4E544BC53F923867085609EC3FF4C9A2FF16DDE63F0E303693EADE12C0B9F1CFAA2CEFF4BF1C733B5F0C39D8BF8F6D337D02CFF33F0FFC0D461BE914C08A78397E0273EF3FAEE05D44BFAF05405B35E223CBE90A40D7EC8B8FAC9013C0E36CE9C8155CFC3FA6B422E591DE00408BAC6884B915D53F084BA3E65441ED3F632AE160742B11C0CEBA085342E4FFBF81FD7EEB66BCC53F7C74B1000880EFBF64D9B1EC6AD4F8BFF2FBCC3B222D0BC0F463C7B8BB750AC057002D88D7021C403E5D470E9E790540D66459EC62B51740BCD137DD99280AC0E0BBBE87832BED3F2AB8E29D9C990440005770B22DE70DC019B7D479196D0840DFCDDB0BED4917C019EB3318D921F53F8F8D04091C271740FE41F64AA8141D407B315A669393F53FC10108788FFA1440809DDE86EB56DC3F92927F024F01F43FEC4BF7E5453BF93F9FAAC4D4671EF1BF8A28502C028BF9BF4107168A885D03C0F8D258EF0F770A404210AB98031EE93FC9E898D44D6D12400E87B10790B6ADBF321A8611FAB6E8BF0CA567DC291A18401E5331EDC6FDF53FEAD01290D112FEBF621B48AAD0EFFE3FC0D5B48FA8521840E80B4BE24B9A0940ACE8F9B37F2312403C184F06CBEB1140F3C5201F48FF17C04AF47DFC3E27D93F9E9BD677DB3206C0AC286F36F354F73F5527560535B9A83F1A165688D1E1F93F1AFFC89705CDF33FFA7DD4A67A79FC3FC58532413F9D03C046BCFE83ABA1E8BF30FAD1E90DB61540A70092B075B5F63F63E6BD4406910CC08E38335E365EE3BF064FCC9CB66C024072075FCFCF47DC3F962C1680461501407C6BFB818359FD3F7491F3F18000E03FDC2628FE2E0DEE3FB1DCFA676CF910400C1CA775B08C02C0140F1480B8630A403D87EE275B4409C09884F099B6BF14405209E2FA0135F1BFBA7DC445AAF9F8BF3C11FF5FBD84124050A89B1F74CDFBBF0DBD94F17F9F0A402A42EC6F8C261140752C82C6F15C1140BD611AC278DA02C0D0AAFF537D54E53F4EE64D73954E0740E8C8499F338D0A40B0579971CC3BE8BF8C6CA068BDC212C0468E71A43FEF0140B8BDAB8FCC06F03F07F507068B2D17C0EC45ED6BF79D0640509DF526B218F83F85BDB414E5F6F53FFBA760876C27FDBFFAA75B2D675E0D40629D9B20C73FC1BFF15F0F8C052E0F4074E003F6A6961140DFAAFC1B16250940BE8603E2AE1602C03B549663118BFCBF36BEABBFB2C5FFBFFAE5A4712218F13F47453CA3E35DFCBF78B04C721211F93F2F19D3468FA00340AA59966D30F5E63FA1B490C5F9110640120199FF49101240563BBE6FB0DE14C09B32B0DF16910340DC71CF34AC04E33FAC9A7F16F3A3FF3F04C810EDD2791740C80CD9EAE752BD3FF205879441F40DC09B1B5205B86F0240BEE0EB798000E2BF67EEFAC7E96C00C0147662C30D80B53F7B540F36DC0E18C0E89AF0503E0E01400C5EC558F209F83FF324EC5BE95F0D40FC045E1F17CBE53F3AB1BA34B7490640E02053FE036C194005D31EDE15BFE03FC1AA60A67776F63F40379CDC412310C0F40E8F2F714A1940F8A2FFF85CE9EB3FEA452D6486A1C6BFA8F47F7162B012C001142E4C846C14405483D1E05D49E6BF62088E46DEDCDB3F224126A50D3ADABFE0B3AF841B12AFBFE475E75991BAFBBFEA89F417B1540440587C43C76938074038737372993A0FC0F19AD1FFBCD1E6BF473B2F178424EFBFB223D31F1A6503406E7A299432590EC0039A00E4E79CF1BF659CFAC3A00003406FC3F67BAFCED4BF92B5EC96E7E0FE3F6A6ABEE0CDF2E0BFAABDA7CFB456D0BF34929A6350DBFBBF6C1AE4E969AC0940F6D2E49E2915FFBF1DAF1C86BFB6C63F28CABE69E40908C0024C270DE8AD2140164A8818EB1D08C04C0F6D8D605B0640BEB3761D933EC13F083FF7B677EB0C407A6655A1055507C0B0958711921912C0BE0E29B78404FB3F8FDFACE460320EC014C82D12EFFDFC3F839A392F88E9D43F649A860501DEDEBF964DF246E4B4FABFE6031F0A3813014049962EDB328D0DC0B2593C92F5B9FDBF975411260A4700407541728B7002F3BFA2D796AFA050D23F5032FECAE1130540E40478A7DCFC00C0624560148E9FDE3F08E8E8DF327B0140B85C55019C1B01403986A278DBBC01404241C404632AE2BFA2A8102C5369DCBF2C69A30B8DA6FDBF6AC155D2227B1D40D6D2C6803F63FF3FA882A49F450A9FBF74BB9DB242D709C06E9D488286C0F1BF7441507753D4EBBFC092DD8125CA04C0D79E6113612BE53F6676FFC43609DB3FFFC64A2A0EA8F0BF7964BC5974610BC03AB1BD502FFB08C060E5DD77ED4CB93FE11462B7CA9322C008A19A98E51DB5BFB0E6C116A4E9FB3F063451CCC5BA06C014B0A07DC01803C08EEA52860FDBE5BF2A21072915B1F73F7DA29C14091A134090902955A139FC3FF99745072C9E0E4059CAAB8B938E10402C952110692E00C00C13E556D016FA3FADAD680CFFE207C094CDCB058050B03FC17C1C5AB8730440A12BADC85A73D13FFA010071028FEE3F4AAD598C65DB02C0D7868D73A8D9F7BF2E67E3C9321102C034483366E813F7BFD74F88E159F20F40113448BF161906C06E8A696839F6F63F2A1946F84CF610C08A67ABD94B5DF73F8026D578B64BFEBF61CFE6E4DED8F2BF47C589A093BEF2BFEE4C2DD62887E33F80FA03139B690340E66D8A394E4B11C0815090B01D71B9BFF7D4B6716FD1FEBFA7E767DD1113C63F92D6B7D0A646F0BF60312AACCE71AB3F5CE2E709F9F20340EC795627679A104026DCE91AA3CBF3BF69FA2A2F97E303C0C09E3437E054EF3FF008EB5A88EBDA3F7BB72F4BB6C011C0EDD445E4A7F7E63F08553988EDDEB2BF4EB79A27A4DD08C05C1439D781A313C0BA9451A2BFE5EABFE8220089B9A9FE3FEC70EED781BC1440CDC1C778E2ECF3BF0280B7E33015F7BFF27FA126745CEDBF74F437C64FB1F2BF9330781607E30A40D18A9B666A7DF03F58293644D04FD53F8F507D3CFC72F53F13DA2726E40B02404607DB31DEF10040C8A22DCA3E64D43FD4E15AAD0224FCBF86E2BF40DF92F13F189104CF8D51C83FE54DE205B2DBFD3F415681CC6462E03F3D184667B8AEED3FDB19AA402C4206404CF4E42BE75E0DC00F7395DA510E044006C9411026A217C0645BC0A9903F12409E7A1709F8C7E0BFC40E35CBFCD308C09494CBD51994FF3F733C07DD581D0340179A120629F7DBBFD91B27D3CE0419C0F746C89B7E70FD3F7C0F3301C6A8D7BF97B2B109CC31F5BF7AD980DE82DC00C0D98EA37875810E40993D81AE3B6F0140828E9DD2ED38F7BF0151D0EE292EE13FACAF196072A8F93FA4F6D2E60E21F93F788284320F5CD23FB5AB5C6824B5F63FC0AF0A41EA43F7BFC0063271AC7D10C02A0EAC8E60CBEB3F458CF774442CF8BF944578B4E943D83FB81B6D8E75E310C0C6D9A642FA4F0A401EF01592408DFE3F67069149AA6FD53F3A80E5B474FB1540F38951C0038C16C0F4C57B4C7999E13F26170BE6E8D8CCBF3B8F30017CF703C0FACCCF336C8105C0C2F4654B90D50B403B205D98B5A40040F8DCA8BA5129FE3FD10C6369F19B01404F7218AB1FD5FF3F8A6B1130F93AC7BF3E32D99763A7FFBF6C5E5C4D8795014006ED1AD2D020DA3F8C456A448676074036318D093374074078CD3446478413C05BBB9B9F30E701C0F7DB7C9F00A915404FB94E946998F73F20DD4CD663D80C407E0294F6C17B09C098351BEFA87AD53FB5B3F28111D6CCBF20C440F9D73D05409D5CF6B17029C83F5E05380CE150F8BFE8DD933FC1261B40292977262F8AF33F6E5DEA3F2D710E406CE6963EAB9801403B28EFDCBD92E7BFB83EFF51C69DF6BF5C9FB07589CD09C03E0B906A05020840DD2D20EAC10007C0BB86D8B61F050940FC4D55186FED1C400A72CCE145B4C03F2E18B63B1DDC1C40866C006C668913C018F6D3AD71F1FD3F270613BBD584F7BFBCBC89CC94520A406D661825ECA7E03FA15AC1E483DD14C078B468ABC3B4EFBFD550D0381FD709C04A3058D7B6491A404E088544E1C0DABFBB7C9F723383FABF8CA0FFEFDDAF074006A9AEBCCFED07C0FC5274DF02E6D03F50D2EE61D337FB3F0015D1A213EC04C0F0DDA7FCFFFA1840B05DB5F0A318FA3F18F92A00190311C040C61947DE2000C0D8F9240B514B18C0F44B55CD73ADF6BFDA704D74AC0F1B409DB64BAFF730134042609552FAC10EC0C2E2CDD21603F63F4422A182A7A314C086D1E4F6272900C07C72DD73BC3DFD3F283F571B3AFAE7BF5EE396F702E7FC3F7CBD1794F598F9BF5CDE263D7830FABFB2B9E180295306404A9E85FDB4F1ED3FA1615178E36C1140741F6717EE8E14C0AF60171E3497EFBF67A1B4A66E7BF5BFA2426776A9AF0E405E10FA71C8FC0440B0CA0BB539AAD53F717D7FA811E00340547A29CD04C80BC0CAE3F7D855CEE0BF7A7537EAC8A6F93F36628E0C289500C0E1E5D609934B0740F09F1CEDD8C4DFBF6FE9B5F3DC0501404DE2DC1440E903C05455C8A886EFFE3FD5815F7F021801402A141171D804FFBF869E09910047B6BF5E3B6561AB54D13F80A0FAE8C9050FC025898D99CDB7F6BFC3AB515E8A9C10C01AB3CEA3E959CA3FB2E53859CBCEE53F3E33EA0342B306402C47F25F835A06407976C5F241ED09404776FAFB623FA93F4547AC337765B73FA8AA38DE213CFEBF192BB3202FFDEBBFEA848260C221FDBF8A3A1F8F0E6D04C0A8C08D40B4BF0040786E77B1333A19C0A74686247154124085CD9ED840B7FC3F354C007332C8194058D6E04D31CF16403E45F558DC49FC3F88E1575EB20A1840D22B361874C713C00B6E1A0CE61EE9BF9C5E5964C6EDFC3F227C720D698CF73FC62902C0A8C209407E76319BE3E9F0BF4AB04CACB767F3BF6F64BAA521FCAA3FE387F8EF95DF114042E917F41D27EA3FF0A6DA2FB13806C0EF90F8CC2AA51340BFECCE1B4B440340B8A2DEB4774DF23F0728D17EFE33F2BF0092BCFE860AFABFD6A518292F65D43F10A407F3850AF23FE72854C3A7F300C0F6F24A908393E13FE49F5513450800406F8FB58A574CB23F4DE475C65901F43FC0B17295564102C019B8A7E2A54FFBBF13BEF2D19958ED3F51C26AF81BF61740A897E5AC1B2806C03D9C5A7BE404164070425867D67DFEBF9E6E280CAD19F73FE2F4AA804D190940CC16CA994C0D01C0095EE72DDAEE03C0863E8ECEC1FDD8BF1B8E2525F0FE11C0208B3C093DB00340F610B3785D8D11C0C4A0EEC8C94208405CA40D899E771C4091A9E7899A6312C0F812BF298057FABF465071FD7F1816406D5074936C88054076CBA7C4172207C018EB5FA7AF43C2BF6FFB3E25EF88ECBF64EB8A159830064044AEC5CCA7BFE23F065E2DB18B160440786E3627252D13C04AB084DCBB6B02C007CF50F5A109C4BF3ACF41A0AA300BC096AE361B1AD80DC03018BD3C044DF83F1E24DA49C51C0440CAF4F7667F2B0840E280E4BEC265154098770C15FFD502C0C02953AE209412C0DE5B658D5CCF08C087FDB64B53E6E73F39F5545E7742E7BFDEC267EAEF1DE8BF9CF03783E5F200C0DFEF0735854414405DA548FF9C5FC3BFA547044ECA6BEEBF586385A6226CE0BF0C4F20C7527C0CC05041984A9EA20BC00DE37EBF8BB5D0BF464635338837FABF7CAB5D19A852F4BFE6F907D2133603403EE025F2E6E102C07E375B3A8BEDF83F22DCCD715510EA3F25E4B1D7DB21C4BFE0CA33B5C6A0A0BF6C7234C8D1A706402C21338E547512C08D34476DFDE8E5BFAAAA833E0AABB43FDEDFD8EE55EA03C0BDC06C2DBC0D04C07CC2C6BF0A33D73F2C9937E2CB399BBF2B9BA141FD7803C0CE687B9185FF13C010B1BECA048A1CC0F538B6EA67201840D6E4B7D7532D11409348552FF12B04C04D45B180DB9BF6BF6AE053E555B8FA3F343C6AD82332DFBF3F19A5EAF2C81540C5996F91AD91F33F77E73388AB79F0BF321DBA17E95CFFBFE49CCC95D8E5EDBFA0416F8DFD6504C05B7976C878C40B40478AE723BA041640DA5389DDED3AF63F7821368A57631840B2A82A8191AEFABFB99D64211177F0BF6D9299122570E43F4640747DD212E3BFB12F1A2937820FC01CFEA9E237FF06409305EF743E8B0B407EE91924824CF5BF484B76C34F6708C0B2196337A5B009C073D51D5313A302400EF53066B532FF3FDBDB5A61F349124043DEE655C97213402FFCC952993000C0D1BCD7689D1907C0A11CB46F30EC074048509D3AF1D7F2BF314F201622DFE5BF1C0711F4062BD53FFA97C473817207C0B6EC9A640BD51E400A8BDC09A076D93F386250043E21F3BF7BB221AD425E0AC0C2D8FE21231CD83F0B7C2029E2E2F3BF8973436A02A90140F43B83160E591FC08486B6164272DF3FD48C4B6EFD7C11C05748D180DF0705C0042E828AF82E09C063FCE3FDB66DF5BF7AC296FB41101CC0CEF98F8E4307DB3F53130EA5991B12C0847F44774021F8BF82395BA0E9831D4019DC85F6593CA6BFE415C7CCF88511C0AC84ACEAFCDEDDBFC219236F444604C0A807FDC66D1B0A40361B3A840BFBE73F68D30D3157D3EFBFCD93D704BE310E40D4624F1E3C48E2BF87870E4C555EF4BF4EF7223409FE124090CD7601B1570940123398090B4E06C0A48DC5F84584F03F94F76D6DBFA70CC08AE71AE07A401240C83838F9389002407DDEB0E4BF92F63FF49AEAC6D000EE3FB185D3020E5CECBFEF9DD6DA82171140C64AB0392F53EBBF989B70FD1BA70940C6EF20D452690540BE1B8F50B5291540A8D8D31DFA4DFA3FF2C2D6A61EF509C05F059CAA4ED9D33F72D1F306B3C819C070F06CA638DF04C0629E35A18D11FEBFD59D57DE7A6DD13F19F290CDAF45FFBF102FA455DAAB06C04129DD836D00C8BFE836CE677EE81040BC20D1FE14D20DC02C04CFB8E05203402C30B200E138E1BF906313F4A55E0F406A21DE3C61D6F3BF08913FB20616FFBFDEBFBAF22AF61AC07389DBBADB26C13F5A35600E0FED05401EEF80F6B649F5BF429ED5FF6C2713C0578A4DBE809913C0A62416D91D21F23F3480065E8943AC3FB3C6760C0A13FFBFE60EB0AE2928D1BFCB4BAD5009CA0AC01A53E37CF40F0BC07DDF30AE4618D4BF0E7F289E888201C0BF4748BE0CF611C0E1C514D203B20040AA1AEF0627DAFCBFEA1A921F0D68FDBF48D24817CFD2114079DFE06B976FAD3F30F777CE67DF1B401BE1D2B428FBFBBF0C0FC3D614710E404B6ADF716AA3EBBF74B8B192FDCB12C0A0158109AAA114C0127141CBA1C10140969DC71B29990540702E784C59CE13408417DF99AFE503C0CC4B6AF8E0D6EB3F31745821564CE13F055326EC4AC0BABF60E2B0532E89B43FBB9591927589E13F21D335C1991E0740987384D205EAF7BFBB4409E23F4F20C0AB2B8713E91EFA3FE2B7385B578904406CD249D7A4FF0E406CF15BAA481BC03FB60942AB48540BC084AE16A7027CD9BF5AE76176A79CEEBFF46C2E8556691340854ED6691B130EC08DD85714E2C7F63F7AE5511E84E10A4056BEF5255ABF13403383EB25EC1CFA3F00FBF159473CFABF595341CE68AAF4BF32D5118C543E1FC0999793880EF4F93FAC9C51C337B6D3BF80223C1E0C6C00C05CDA454EE7580E4002EE4B50A90BF53F28E18C17DA050BC0BCE4DB73630006C08793635BB2430340169E0511F9C2E1BFD8BB7DB68967D93F38D791BA38BD0E406290A990C48216C0B5704A43ADF701C039980047CFF709C09A1AFD410CFA1EC0F84D0814D1BA03C014EE56263DB00E40F6363F61E4C2DDBFEC28E1471FDB1640795116B5A0FDE6BFF6495EDD86F0F13FD04F8467FC95C7BFE67A071C069BF5BF3C164C9503A60FC0688FD42FC796C8BF10A16A2348DA0F408F1583CF0906014032E413E187AA08C00493ED663D2508C00CFE6287D69202C023313173911007C049C1E3903D89D6BFAC0BA83B4415FA3FDAC7C96BE2C40340142F78BC0FCCFCBFE448F95B0B6EDFBF7A70627CE212F53FF8C742FE44D10F40DB1CAA974F5EF2BF6A00C7EBCA9706403BEE6FDBC7C511C052FF7CE170360B40785DFE9E7514E0BF04A2A306C510C73F204963DBC6E5EC3F900A5C711CD508C004B89E0D2A5AFD3FBA7F19F1F54410402C50C2BFF587F93F247CFC2E575BF43F65ABCEC54B94DD3FDEAF513F8D1FE33FE13B65DBE324F6BF0B60DE65076EFABF4A108F85963900C020F14D549408EC3F02360EE9C480E83FEA16603D71D7F53FACC606879275D1BF7D44EBF58FE200403951209BBB48FF3FC53B4D575F2F0540E20CE3AFF885EA3FE7F7700F8F7004C08845DAD0A1F010C0E6DBF30A1BD10240"> : tensor<20x20xcomplex<f64>>
    return %cst : tensor<20x20xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<20x20xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC881B3FD560C3BC075A23D47A8384440C786A384B84BE13FAC27F88F7A310AC0ABE9F3A5A1FDEB3F7F9EABFA1344FA3FC3760F8041D5D1BFCBD5B483800BE33F4F9A3EB92F4FA1BF96B6F3E4A3C6F5BFCC176C06349213C04AD91B67103302C06FE0633F858F7CC0825EE23FAD1F72C019113770EE691140F7306015605BFA3F53D8C9025604D8BF4F1B020940C0F13F3CBFCA760D1B0DC0C9C457CDB57727C0D0D2A20C8AD1CCBFB68E9A1D965BF1BFDD23D9D18398234045C934128D2F23C097A08DC4457FFFBF664831548F6E14C0E5197553A15A61BFEF94F9C17B95F03FE3965C7CF800F03F353A76EECCE1E43FF567E2F8E7B433408B70452497400940078358685A7A3740327722E618D21E4075B88750850336C012893B606ED34140F88D018D9C37B2BF4F44E9B64EFCF1BF5614244E835EFA3F5713FFE5F67A22C06133FA7ECC6C09C07C1B20FB771FFBBFB5718AF33BAB45C04E866C0E0D6720C08A2F7B932E763B40808602C5C0DF12402BD08D500F3B3EC0F6C1C4FC62D40AC0E09BAC6864F3F83F6031B407CBCE4A40F3C4D4609E1900404F99D6BDE15F1040A55D13AC7CF8E93F9F833AA04B6DD8BF25DC08B3B98F17406DF679AB40DCF5BFEFC7DD5706072D408A112E877B1509400160E048DB771EC096F71292ADAC2E409E0E87A22BE1EF3FA3049E908F704840AAE0DD9307E31D409644D229C0FC24C05CEC0ADA5F3F7140ABC5CB0B3D2B4740217B4591BFFEE73F598AA094048BED3F692AA41303072DC04D57FB9A99054BC07AE917048F34C0BFD88135F5EB37F03FBC6568008AD749C018EC5C69E46153401BAF0D091A361DC03A9C53D3B0ADFABF4D261A5383AA2A40D5251A64DD4D50405BCABAA5C4B70E40007EAC334CA2F53F1058F3F10C52DBBFD968070F0827F53F4E4B3C0DB5650CC0233FDEAC9636E43F85B1EB4A865296BF304D377BE964F8BFE4A5718DAB6E2D4059E45A6A88C103405786C14FD6D57EC03495EB1932576E405CC26388F03F67C07F496EB49CFB37409A0E2E7F2E2AECBF81B5658208D2E83FF9CDA34223E93440C2675E1514A9FD3FC4EAC71AEBE344C0A173C6476A736440EA87DB89937D56400E397003C105614096E5D644452BED3FE72CC37A1E57FCBF4DA8E2EEB871C23F29F58734D7B2F03FF01E332DE7CCF13F0EE8A0DD2DB501C064F71DBF3071FC3F96632E29CC21FBBFADCC5E6EC64F23405B0AA01E3A5823409E896FCAA3FE4840D8CEFD9E3A3D07C0ED37D2AFCE58EABFC8A6FE05C06AD5BFF39A8910D1FDE1BF1C9D6F2F220700C05D25104C3E860A40DFB3E1D4C4B5E6BF17D0B910485301C06C41118AC63128C09D18B6CFBD2D45408F89467D2BB92840B6344F22F216D8BF9E657877CFBED8BF78122D52773E00401257A59BACFEBB3F97F2B4968458E93FF03A85A117CE03406D1F71F503C501C07E6809FCC909FFBF5E15CA53B3CCE1BF70B6A2A0B2ABEFBF079873667050FCBFD79228188523ED3F5CE0522E1A82DB3F02E29EF2454AEC3F0D89FCC53C42CFBFF19E310364B7ED3F9C185E97C76A05403CC5522310A1F83F565ABAE1D553DFBF2500351E550BF5BF5C3CBFA951E113409B45AF7E7BAFE9BFCCD302BD847A15C0A803D2CC7EF724C0D3E29EA3221B8ABF9489797D9E2DFABF6BEFB915C40421C070A004A9FC4349C0A3631F14A1F216C0D66E1966456F29C094AF51D441263BC0451602D6ED213BC0A4AB56569D59E6BFC5164B113CD1D13F032605F4990D2440A4DA2971A60123C00258E04626EC40402BD1B1428F5145402FE5E15468A3F03FAC7B80B64136E73F8E65FDCAB16FE13FDC16160BAAEB2040500CC43D7263DDBF44406F66614500C0812F51BEEB743340C4322EBE872105C0603331C7EC8B1EC0899DD50B236F37C0639B6B48A5751DC0A2862F9144E721C006EC890AD7A8F23F3E0E0EAD3A6806C0357D3B16D17DD0BF323B9F790981F9BF67C4812DF84FFCBF430179771667F93F74564EE9A32BE7BFAE7A40BEB153DE3FA880C42D0F2036409A93186231034440C2106D75B6FA1240BC849388B50A0A40AF30017BD8F20940A4BFEFDD46B5F7BF255C4FB3E535B8BFBAD366DA1A42E23F45E1D6BF42C81040ACC5442CCD9605C08BE70770998C0EC02202EED74643D53F15C1B035843A5B40D2F85323DBA86540286CE5572678FDBFD5921ED3C313F3BF04B1C3B2D40BE6BFD9CDF98BE0A6DB3FB3888CDF8B296F4006BE6FBDF6FA6140E2871487EA31F3BF93E302B9A8E5FA3FF75B8ED92E646640AF1E6DC975A96A405357A16F33367D3F0713B910D63CF03F234314D0B8A44F40A9F788A2B5764AC09312EA08EC5EDA3F28701CB727F4DBBFF6C9D80CD416843F79F7C0DB4BA3EFBF6161FED6BF8418C05AD756C52233F83FA6D9F0BC0EBE32C0526397B8663930C03E19FD5DBB5AEB3FDA192A9241CEEF3F14ED1E84951A24C07034CA4726CF33C08C9BFCD0A4341440D238F186F3B7FBBF0AD0CA516E4707406162F27E156EFCBF7EA4D6A70D63A63F502B79F10B49F0BFE57A5EA6A0F611C0A41DDEB6B11A27C03232F5AABB9EC6BF149797A7FFB9C1BF7C47311913BFAAC0781FCEE362517BC0CC66F0DDC72520401026D1C014A4F13FF5CB7EF3F21732C03015599E3A9810C0B41A746857B11540BF76106684EA464002B150A208CF14409CAA71099E2B3540781794CEB4DBD23F03CBCA387E48DFBFD3B1EC2DFCE0F53F9919FFF09F9502408E51D3E948C11640092833C86A4A33C02B3ABF627A76F63FEF7870A0F4DE0CC02954E443433BD0BF48390F4ECF2CE03F417C470C8D420DC09D8DF2BE273CFF3F201F9C65E8E102C057C027D1545E0E401CA8EBD95F990E40A4A1FF9835FB03C08EF0EE4C8D5BC03F82E95BD5D2E7F0BF839392D3FAE872C007CC5D46E1F08640073C421560EC9E3F415346CB02A4B63F3A3CA43855E6EBBFC2EB139B1E8EF4BFC48D164E221A15C09CAFFBBECC9C104089E111A8F024CC3F3EF5660B441DEEBFBA34DCBF449C2E40BCAE37710E99D2BFE31A2426771BB9BF91FA4DC46A74C1BF249A9AFFB53C8D3F4DBFAF2221A0EF3F84069D73E2E2184030256714E18D17C0C4A7A59033FAB0BF7C5541F96FCCF33F0CF6EE6B48C526C06B71E19C22174D4038B55713EFE528C0F60E6A20714E33C02520F2374564CC3F613EB4AD3AB50E4064B00F2E66B623C070E3715BE63AE43F6C133CCFF1AD18404AD7FDFE51F0FB3FBB9AB191A011E9BF576EFC1B60DCF0BF7027331E6C66F53FF795966C3ED7FCBFCD7C24B6280CF53F035B3C1F99E3FABF64F9694C5911F1BFD6838E6F43811F40778FB69F83A10EC0D7D13FB93F424140B29B1348CDE1F3BF960F69B7271A09C0459B79A1C71FF3BF4670E3E5FA36F03FB010BF215A0801C078FA623EFB1C1540C80E52285ABBA13FD3FF8F463426EEBF23F9AB801F53B73FB445AC8788A0EBBF49CEEDF880E4A5BF945B918F6F57E33F7C52ACE3FCC724400C1972EAE4FF3DC08572BDBC9A9E0AC0F4BF795322151440ACD25DDB2706BEBF55F234A1F8C8F03FFB0BF34340ECE83FAE331D8504ECB7BF66AA869F3C8301C0A1372336ACFD254070D7302FEB72D43FC34D428231B1F43F60D3917BB7853C40D18FB4489A2155C052099C59926DF3BF5666EF05EA62FCBFF33E1F462CB0F63F4BF24CFA2E8DD8BF3E74812C417EF23F6700FEF95286E03F10A504563204F2BF16B603C4824BF93FA513619D60210F40568A1AFCCF23F53F952627BBE481F4BF3BA268F0A14A0540CC65683DDB7BACBF0A592A662F2BEF3FB693B7411B86D43FC96C4C6A2309ED3F4CE7667EE1C61BC04835E884FE6310405878D69F3EAE1640DC6521FCEA1502402738D46DB6B9444025A6579155FB37C00C3E3C7F766E1140177E0766B0882440427825ACC5981340D7180A87DC9D02C0CB01187AE34C5140646F1D13665B6F40F52797509396B7BF77F6F5551997F0BF6AE6595E056309401FCFBF24E5C804C03291E885E2A1E03F89EEBBAC7AC711C043D72DF89EEC92BFDB4E1C891C5AF23F5A88D0F858A7014065EAF2B4E5B6E63FDFDB100FAD01CD3F3650A801596201C0150E420F19EE33C0F278459E5E913740B93A6C20AD0300C0C7B76636E022EC3F24965B1523DB40401AE5887CDE0914C0F9C179AE3EF008403B3C20C19B23F23F6339F629714E5840BE2EF503BF575240AEE096F04D08E23FDCA4BFFC1C85D0BF0EA7278206AA1540C37D928B936605C0E9353CD4439E1FC0121FCFCF67572C404ACF5F691C58FEBF09A9D5A85CC6054025105026E33A0C4044F8D4BCEE82E5BF3794D14D83A600404DAFE904CBDD0740B35E0019F146DABF3CE367FD7EC3CC3FB118C89065D8F83F2DB720556C8D22407C6AACE958F407C0FE47E78C3EF00CC021EEBA35B39FFDBF384005595E77F0BF040939D929CA26C035738C76E2E70F408217D34E43B5C93FF4BD2A42BFADDE3F33BFC6B029D8833F49DAF787C243F0BF74566A4A8AFC6240062773B9E00A7A4071BDC23F776A2AC07624B3B0432E3240CB7CDA903C24C0BF31863B92AA49F4BFD8D2690CFDDF284027AFD2646641FC3F2580983580AC2140DCD8159F8A0AC03F8A4F66C12E6C8540AF783CD5487E5640B56CAF6D7E0D5D40D1B3C5C566EE8440ED5A4274F270D43F42E0406D9B780AC03406437568432740A729331E87BE1A40069F378B1B3E49C031C0F9D8234453C0FF87E76B1E3228C04DA2F8CD5AD80C40993ADBECB3F9A23F3A2F8C8D8B5AF1BFC00C59A69D0B23C0F1CB2AFCB539F7BF92E35BED42C6A1BFB6BB913AA76BF03F31A1287B052D1BC04C29C9031EBDD0BF6BC9EF0B7E6DF1BF5A171773FC0403404FD73AB115DB0CC063A30ACABD71E93F5D7066FB5F87FBBFDB70FCE63A34F03F169E0E00ED1E47C04EBA09E2CE994340CC29BD9D7BA6E93FD01672ACA65EFE3F2EB1830D6618EE3FD4140F753FAE0D409A0C042BEF8AC83F22C174A13622F43FA4EA247CE40FC43FC1F152DE619204C0B30A422F4A4113402F301D0B27541A40C3CCE9964F263040744163A770BE414051DD43B351BAD0BF4FBDC4162DD1F7BF405CCA334A1D34C06CC95499F5F4264021BB3ADA607BD1BFBC7B2A12E4ADE43F9C5A576E60DA2BC08CA76C13CF2B20C0D780CAFDB35AF2BF03F8C0470B2102C0CAB7F9C7CA1F2040552CB6F7EF9111C01CB177651B4E0AC00377F4452FB004C0684EE3C52109FDBF289BF113B8D40740B127894EAA1E0BC0CC6C5E0364BAD3BF48906871D2FFC9BFC9CD4245DE47E63FABBFF87DD489F03FB65E9B1081AFFD3FD1E7A4C22E9CC43F8598820E5396E43FAB6D82313CFF1FC0111D37399F5407409DEABC28397C294005B34F3C822FE43F6B62AD786D619DBF1DC48D248483EEBF5A0225BA645CCF3FEDE260D815D2F5BF96E2CB0DCB810940A7A95BC35766164065E85C5480C24140B6ECD78382FD70C0C659215769190740D94F43990BFBDF3F1CDA07DAE5593DC093ECB446305C62409D595F3BB87847406AB9A8F357CB68400020308D541ECA3FFC04626E4F98F43F7383C44B867500C0F9FF5E888701C7BF6083D7BC725FDCBFDB74E7BF072BF8BF3FCD40ABE4178ABF60D74E184E18EFBF3A34AB1CD040EBBF3D9375FD00D2DEBF8F5F9449FD3949C0A2EE5CF4DCB946405500657E7EF0E23F037BF499571BF9BFF1F0F4DB2F9802C0412B599B0882EA3F46F215BF970CE7BFE3526F26D54BF7BFE5CA37A88A05CFBFAD51D0D692C4F03FF701F8747913973F45A77C8FCD72EE3F84D3CF40E50CE53F1F2C99B7929C13C0F647DE931B1AF03F73A0456BE1CDDABF41A0B2259EA016C09B98602F437D16C0BA8F23B333A8DABFB2E6DACD38450B40B5359C680D7618C0CE34AC9F268D23C0E66062DD9E3016C0B74FC3EB1A8502C05F1F195986784140D5BAAEFC474B3C402C1C67F197FF4340350CC0C6A7821140B15076E25EB251C09C9BCBF67221834010639C8198FCFCBFE62B3A79F2A1FDBF16C991BF6A761CC0228FADF21655FDBF4C3834945705B7BFC078DA279D27E9BFB354B881DC961A401E9F8A2491C81140C9AC15DF31F4DF3FFE9D4834E0B61840FACFA7D8EA8E13C09751C66BDD31E9BF60B55FB991DB28407A0F71DF5FA42040E366EF3EBE27FCBFCD8DD363D87AF63F7536324D025918405939A8524A8820C0F961FB73329BD63F8DAA9F0BE834154029CAD234D64620C07469A58346421E4055044AA1837EE2BF4181640115F1EBBFCB55C20D52C5F6BF7D9625B5CCA80FC018AA2E1F929AB6BF97BD3A0C2457EABF365126962656DF3F733F0EECEB8CDD3FE332D2E362892EC09B3D4837A35910C092846215E477E7BFE1608116E46804C017FDAF9EC4170FC06B510D3744590FC01BC664CCC8EAF83F6E3240B088DCFC3F0EE5BFCE6734C4BF7D3E7689C7D4A0BF2D14B4B6906FEABFA7787D20B5F42040E6EFC8C1E595E7BF97AF9E7455ADB93F21899B9CF6461340027B642033CB0CC0C430A972D1B3D73F6DDDE6B2DE079DBF3A6FA06978A4F9BFF520045F780B1640D5F8D31380FD82C04983E252468563C0C0C1E130B6CF3DC02DD03881F34B35C0C2CB05F20B80C83F5B257585675201406159776DEABAD5BFDCD341A37FA4EABF06FA43AF0F93E93F0FBA355DEC50F9BFC676C98F038C00C0FEB782B1414B07C0FCE381E6BE151840D22A88C37DAA00C0AFE7F026C82C3640B8F980776A395E40D52524FE8D7C35C0D80450EF2CA76BC04E9A18D2D55BEFBF3422AD8DD819EE3F4CA2538C3444DC3FE34145CC6313EB3F433AB4B9B5DD20C04403984B441B05C0FCD47549C30DFC3FA60ADFB23F97C7BFA2CC0D4F200B2140B1BDAAC852082240668FBCF1B5C8DEBF5780D2E734660CC04798352CD3493CC044C38643F1104DC0601CFB5D40B0214054B35D5D54B4F53FAE9ADDBC023BF2BF76DC262FA9F5F1BF955C69855A14D5BF9050569BEA3ECCBF6F3C6B639F089040703120555AF47A405294543D1ABDF73F065B17168DC2D13F946D35149BCCBF3FD4384A81303CF0BFF5D62EEFF98FB33FD3F28F61546712C0316DC3222CD9C5BF51C753A59AF0F03F7A99479713921B404CC41193F29DA63FE0924A862B23F5BF4E0F39C0BBD2F5BF2DE91A27FF87B4BFFB21FD752626F13F8EBF2331B573EFBF00BF8BB0A4DB00409903290121F68C3F51B60E592449EE3FEAA6F5C4366ED93FEB99A60C2549E4BF08043A4B921B23401638136DCAD42140C2ADE42E4E1CEE3FC77AFF6CE913EDBF6662577026A4C6BFD256E66562DCF1BF372FB74520D44CC0078F589ADA33F8BFCCD1045ED39A10C0E3168A35AE011C405DDA7E2BAE62054059BE58CFD7C831C001389524BCB1E93F2020B1D43E4B144062F8CDD55CE0E53FB30888FC2141F2BF89B0AEE0C08F3740890E36FC230B3BC07399B5FAE80826C03C218F6BED3016407DC3C93F5B0C1DC0D9687A01C3BE5840C0D63AAC856428C06B732B29585D0F40C23E0A8B19F770400D866DD4C6FF63C0B6D3A9074CA408C0CF270DBAA4D6EC3F0CE7C1652F600A401EC4E5CCFE72F1BFAE31B0989086B63F95D91C11B4CDECBF042E817C84122F40969E0FA87E9F2B404943599927B2D93F04E68E7674D5E9BF1D015007006AE23FB9E8DD23F6E7FBBF93A01BE081327AC075316312A6404C4098815A26C062FD3F48F7FE19603B1EC064321A17306426C06DD2D21A85814D400B8960EF463BF63FC8A72F2C5E34B83FFC7115E46D580AC0AD0E3E919C29EEBFEB1F1CCE309B2B404588790FED3A0B40A97E75BA2DABC73FAAFD99B1EB60EBBF955078F0A9FE3540E95F7205D162434081C838038BEFE83FD46D824A670D08C0BCC3860B877E4540E2A1454C92CC03401F93EE7A8C8357C0CA7792C3F45580C05E7CA13E382D2D408EAAF60EE01631C07C79FC64D1AA37C0805E87A95CC948403DD124D8477110C09C23176117DAFF3F9D5CC9FBE80A4CC04880AC7D788445C0A66953F63BFFEA3FDCE47898631AE73FF2D2AB47B3B6BABF94519F2A3EA0B43F4117AED7FDD8E1BF3B23B626CB63D23F292B5B749611E43F68E885F021E301C05B53281C748600C00DD910887118F73F127A015BECE237406F3CF6E0AF3108408A4AC00E110A2CC0ABB060102DA417C0EC420B8BD8E1C3BFCCA8412046A9F7BFE11CCBA3BA2109C0BF394ACF6A3D35401C8850940D9B094013029E2A9E1A2CC064E3024AF1ADC5BF90647817763005C0E3456CB68E8EB2BF98CEDBFE4A47FFBF4D8AAC5E738C02403CC6E2CBBF84E9BF5DD9ADBE815C0840F57304EA6C3603C0443AF67191E4FABF1DB10DF767EBDD3F98EA42C9022617401C5E29BBFB0815400CE9473C173CE1BF69B0DED61AA7DC3FA952F098317A3240EF00BAABB3702C4099BE3CA8C895124075BD4914BCDDDF3FC647B304B2258C40E3A221BF098C86C0C06A9B9F74B33440FD2572C29BCA24C055328A911D855C40C3960D33DAEF58C0F7FD7EE70092F53F5CE9D3D103E7D3BF7A08979F14C3F33FA127793932F6F73F380BF32C7783C03F0ECA41FFF644E8BFD9618B8C678710C0403F2E6203C6CFBF1A6B65AA4AD81B40CF1B239178FA1DC06E0E9C5C6EB820C051FC18F9BCBA08C033DB6E4556C5FEBF1BF6FE124659FA3FE7896D5873C004C02A7CA9BA7471F7BFFACA5EA76497F2BF3B6BA6E0BEBCF7BFC268C677588BF53F3D803005E753E13FB1BF61BC0B8B44403ADB858F65DD25C001DFE7FC667EE0BFBDE36F170FE6C93FDBFDBE108E7BF0BFA3914C4E2898ABBF2EA8C1600A5AFDBF4D2DF4F89E8604C0D419B2C25D40E63F289AC1AF3EA103401925ADE52C55D93F04DACF4770F6E33F337561915D62C33F2B633148F6E800C079A1C28CD41F03C0ACD2242CE3C107402252A1F58F08C63F6C14021CB37BF43FCACA5E29D52BC23F966AB09F2B7AEC3FE07F6E617E6408C08E2671CBB34FFB3FBF69B25DD4B4E8BF44674A0AB127E8BFA740B7E4C34F3840ED9B87258B853840"> : tensor<20x20xcomplex<f64>>
    return %cst : tensor<20x20xcomplex<f64>>
  }
}
