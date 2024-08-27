// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<100x100xi1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<100x100xi8>
    %1 = call @expected() : () -> tensor<100x100xi1>
    %c = stablehlo.constant dense<0> : tensor<i8>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i8>) -> tensor<100x100xi8>
    %3 = stablehlo.compare  NE, %0, %2,  SIGNED : (tensor<100x100xi8>, tensor<100x100xi8>) -> tensor<100x100xi1>
    %4 = stablehlo.convert %3 : tensor<100x100xi1>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<100x100xi1>, tensor<100x100xi1>) -> ()
    return %4 : tensor<100x100xi1>
  }
  func.func private @inputs() -> (tensor<100x100xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00FDFDFE0002FAFE00FE01030003FDFB0200FE010001000302FF04FFFD00020101FE000200FEFEFEFA0003FE02FE01FC000600030003030201FFFF040002FFFF00FDFC020000FFFF0000FDFCFE0400FF0003FDFF000000FB00FEFD0302040201FC02FC00FE0100000102000101FA010001030000FCFC02FE0203FE00030100FE05010000FE04FD02FD00010000000200FEFF0003FC00FC0000FE01FFFC0200FCFF04F903FD04FEFF00FF03FF03FF0201000101FEFB02FE00030100FEFEFE0702FE0302FFFD0000010006FF0504FF00FCFE00060008FF0200030000FE02040301000201FD00FC05FEFE02010000050502FF01FF0000FD0300020202FCFEFC02FDFE04FEFE04040201030105FE0400FF00FFFF0302000202FFFFFFFB0000FF0700000104040000000206030400FD03FFFF03FDFCFE0101040000FF04040001FFFA02FE01FE03000001FF0000FC00FD01FC0300020402000300030001FF0004FD02FD0500010200FD02FF0000FD0400010101FC070300FE05FC01FC0000FC01FEFF000100FF01FEFA0601FD00FAFDFB01FC010301000000FE02FF0202000500020103000502FC03FFFDFFFEFE00FFFE0009FDFC0300010000FA02FEFE0100FB00FD04000100020201FC000000020204030201FD00FF00FD00FF00FD00FE000000FFFE02020200FC00010002FD0000FF00FE0201020505FF0301FEFE03FE03FF02FEFEFF0000040202FA0101FFFEFC040303FCFF01030502F900FF0000020101FFFE000201000102FDFB010101010001FFF80402FDFE0101FC000001FE03FE0100FDFF0106FE0104FA0001FF0101FEFFFDFE010301FFFC0200FD000402FE02FC00FFFE0000FFFEFF040001000000FF0004010101FCFF000000FDFBFDFDFF050200000002FBFA00FBFF0001FD0200FC02FE0103FD00050102000100FF02F7FEFE01FE01000004FD01000103FFFFFEFEFD0200FFFCFC01FDFE0001FF00FD00FB02030200FA01FFFD000306FBFF000000FFFE000302FEFD00000100030003FC010002FF01FE030000FFFEFDFDFCFD01FCFE00FCFB05FE0101F9FE0006030200FE0304FB00FFFC000002040203020201FFFDFFFE0406FE00FB02FE050001FCF9FEFE030803FEFF03FE00010100030200010102FFFFFD05FE01FFFF020002FDFFFC030403000000FEFC010102FF0005030000010100FFFC03FC0201FE0000FE0401FE0300FF02FEFE000005FC00FF03FEFFFE0200040102FD0001000101FF0001FCFBFD00FD0601FD0004FFFBFD03010000FDFD020103030104FCFEFEFEFEFF020003FFFFFDFEFF0002020000FC060002FDFEFEFF0005FAFD02FE0000000401FF03020800FE00FD010000FC0701FC00FDFC000202FF0400FFFD0301FD00FB01FC0003FD00FF01FC040202000201FEFC000101000000FBFFFD00FB000000FE00FE000400FEFC03FF010101FFFEFE0200000301FF03FD00FEFFFF0100FE01FD00FDFC02FB0004020000FFFEFE000103FDFEFDFEFDFF01FF000101FFF800FFFE0000FF020004FC030000FF000300FE00FF020101010500FA00FF0000FF070002FDFEFE00F90105030101050103FE01FB0101FF0201FFFFFFFBFD000002000102FD020400000000FB000401FDFBFC0002FEFB03FF000202FFFBFEFF0401FF00FD03050200FE010302010302FC020402FFFFFBFD010200FF01020004000002000300FD00FD0000000101FF00FEFF0104FF040002FE0002040000000000030106FDFFFF0002FDFD00FBFE04FD03FC0000000000010003FA020100000001FEFFFD03000201FEFA020000FD00FF02000405000102030001010200FDFF00FD00FD00FE0003020000000603FDF9FD0106FF00000107020000FC00FF0000FC0200FFFFFEFFFE02030106000104FC0008FEFF02000109FD0203FC02FC030201FEFFFD000100FC020500000001FF030202000000010300010001FEFB060001000AFE04FE0100FFFC0403FDFFFF01FF0100030000FE0006FE01040102040000000500FCFD03FD030000FC01FB03030000FFFF0101FF04FF00040001000303FE0001FD0000020000FB01FE000103FC01000400000302000000FAFE08FEFEFAFE02F9000602000000030404030400000203000000FD05FFFEFF02FFFB00FE060002FBFFFD00000101FCFFFEFE03FFFC0004FE00FC00FEFEFF020300FF01FF0202000000FFF900FC00030001030700010100FD000001FDFFFFFF0203FEFCFD0000FE0105010100FF0000FF00FE00010000FF01010000FC0001FDFF050505FE0201FEFD01010100FD0302FDFCFC000100FD01FEFCFEFF0000FFFBFF00FF0300FF00FC000200010500040503FD000001FF010103030100FC05FB0300FEFF01000700FF0202FC0303FC0200FFFEF9FD00FE00000001FF000003FE0005000000FC00FC0001000002FE000403030402FE00FFFE04FEFB02020200030401FF03FCFBFCFF01FD0100000303FFFFF80004050100FFFF00020005000002060100FCFF00F902000300FC000005FF03FE0001FC000004FF01FDFD04FF00FC03FDFF00010000000200000000030101080000FD000100FF00030000FF0007FE05FF0503FFFFFCFCFBFA0002FF010000FFFF0006FF030103FE0302050105000002F9FE00FEFE0104FEFF0001FEFE04FDFF030302000001FDFD0200FFFB02000000FF0003010100FFFF0200FE0100040004FE00FE00FE00030304FDF903FD000500FB03FFF7FEFD0000000002FA00FF0300040201000001000602FEFC000200010200000106FF02FC00FEFD0001FF03FF000304FFF9FF040100070302010000FCFDFBFEFD02FBFF03FCFFFDFDFE00000002FEFC0100FE000102FBFE00FBFD020106FD02000306FDFFFDFE020103020300000100FE0003F90101060400FDFC000004030203010000FF0300FEFC00040101FDFFFBFCFFFFFF01FF0004FD03070002FEFE02FD000200FF02FE0301FF04FE02FF01FF050203010002030001FD02FE01FF01F8FF0000FF02FF0005030000FF00FFFE020300FC01FE02070000FF00FDFFFF0100FC000104FC05000400FFFF01FE0001FE00FD0003040100FFFA00020400FE000000FF0303FF02010204000100FFFC000300FF0300000202FE00FCFEFA01FAFD00FB0101FDFFFBFDFF01040002FEFDFDFC040000FC050401000303FE00FE0301FFFD00FFFE0100FE0003FD04F9FEFAFEFF0103FEFF04FF00FEFF000801FFFF00FCFFFDFB02FEFF00020200FFFE03FFFE0001000003FF02FF050001030202FF0001FFFD01FEFDFBFF0300FD000105FF00FF01FBFBFF020103000001FF030002020302000000FDFF01FEFB0000FCFE030000000000FFFC05000106FF00040000FE000400FFFC00FE0004FD020000FF00010102FD01FE0000FE0403010501FE00FA0000F8FF0000FF0303FFFDFA0301FF0300FFFCF8030603040000020000040104F8040702FDFF050802FF000505000100FDFE02FF00050305FF0000FE00FB00FDFEFF01FDFCFBFBFF01FC0100FDFFFF00FFFFFF01FFFBFD0000FFFF00FEFD00FEFF00FFFE010000FEFEFA010100FD00FFFE00010003FD01010101FFFF01000107FB04FD0204FFFA010100FFFAFE0200FD040003020400FF000301FE0001FFFFFEFE05FC0002000301FFFF04000000FD00FE00FE02FF01FEFBFF0003FEFF030102050000000002040302FCFDFEFE02FF01FF0204000004FD020202F8FE000000FE0304FE02FE0001FC01FF0000FF01000102FF0501FBFE02040000FFFEFFFE0405FFFE050100FF00FF01020701FFFF000002000500000300FEFFFF00FD03FFFCFB020000FEFA04020102FC0005FEFC000103FF02050502010200FD0402010401FBFBFC03000300FF030000FE01FB02FE0101FE000002FC000000FFFC00FF00FE060202000401FE01FC00030100FD03030202000101FBFD00FE040202FEFFFD05010001FF05000100000401FC01FE00FB0000FF00030203010602010400000AFC0104FEFCFFFD00FCFCFEFCFC03020200FDF8020000000002FE0100FF0400FAFD0002FF02FC0002FE00FDFFFEFAFCFF00FE00FAFEFB02FFFFFAFAFCFB00FC04FB0001FA03050400010002000000FE000203FF0002FDFFFF0501040004FAFB02FE020407FB000602000004FCFC00FFFD01FE01F903FE000001FFFDFF03FBF8FD0004FD03FF0200FF00020000FE00FD03070005030100FEFE030001000507FEFE0500FDFE000100FBFAFE04FBFDFCFE01FFFB0104FD00FF02010001020200020100000101FD00FEFEFFFCFF01FD0203FD01030000FC000302FE00FBFF030500FEFDFC00FBFE0000020000FC00FF00FD0003FEFFFF0005030001FDFAFE00FDFF00FF000000FE0000FF0005FD0B0300FF00000202FE010001030503010005FFFF03000204000102FF04F9FEFDFCFD02000005000401050002FBFEFAFDFF00FFFCFE0305FF0001020000010006FB00FF00FE0400000000FFFFFF0300FC010502FDFB01FF0302020001FFFFFEFC000100010005FDFDFD02FC04FF01FF0000FF0009FF03050104010000FD02FEFF000403FBFE020202010000FEFD00FE020005FBFEFE000805030500FC0301FB01FC00FEFD02FE000202FF0401FEFFFF00FD07010202FF040002FDFA0204FF01000601FFFF0000FCFD0002050000FF00FE000002FFFE0002FEFC080003FF01FE00FF02FD00FD0000FC0101000700FD0100000000FD0100FEF90000F804000601FE03000001000002FE01FDFD0102010306FAFEFCFEFF08FFFCFF00FE0004FC00FE03FE00FD02000000010000FE0000FC0001F8FF01FDFBFE0600FEFD00010001FDFD050201FF000403010500FC0003050101000303FA0000FFFFFE00000000000000FD0000FE00FDFFFDFF0000FDFE00000000FCFDFA0400FD010000FEFD0000FB0000FFF80004FFFFFFFD00FE0001040101FA020000FE02FFFC05FE030000FC00FF00FF0003F900020104FE00FB00FE0000FB01FDFE00F70005FAFD02FB000300FB00020401FEFF02FDFD0100050200FAFF0104000200FA00FD000105000600FFFE01FF03000300FE0101FE02000200FE06000107050104FBFD000000FF0002FB0300FD0601FC05FBFDFF0104FCFEFE00020104FDFF00FD06FDFF0103FE0200010001FE01050003FF00060101020100000000030600020103FC010100030302FC02FF0100FDFF000500FD040001FDFB06FE00FDFE000503FCFCFEFBFD0001FD000300FEFC0200020403FFFC01FE02FF0301FF0202040103FFFF050100FA0302FE000106FD0700FC00FE00000000FDFD02FE0002020203FDFF00F8000204FF0100FF0101FD020000FFFA0502FBFFFF02FE02F9080200FEFE00FF04FE00FC01FFFEFFFE0404FF00000301030200000202FF0002FA00020000020101FC04FD00FF06FDF8FAFCFC050002000000070001FA01FE03FFFCFA00FDFEFE00FDFB03FE0101000008FF0400000301FBFB00FDFF00FFFE010103FDFD0001FD00000000FF010003FF01FB0101000506FB01FF03FF0000FCF800FF0202FD0201F9FC00F9000300FF00FE0000010102FC00FCFDFF00FFFD00FC0300FC03000300FD00FE02FE00FE0000FF00FD01FF050000FE03FD00FA01FC01FFFAFF00FEFB01030005FEFE00FFFA000001FD01FF0202FF06050102FF03FAFFFF02FF00FAFF0001FCFB000307FE02FE03FE00FE00FDFD050301FF00020001000203FFFFFD00FC05FEFFFA01FF020001FD040304FFFF020404000000FCFE0101FD0100010302FF0300FD0001FD02FDFFFF00FD00FCFAFDFEFD0000FE010600FF01000001FA00FBFD04FBF80004FDFF03FE01FBFD0005030100FCFC02FFFC00FF02FFFC0500FD0204FE010201000202FF03010000FF01010001000000FE010100FD03FF0002010104000001FF01000004FFFE03010200FAFFFFFDFBFB00FFFDFD0000000002010101FE01000201010002FE0000000003020400010001F9000301FFFF00000000FC0601FFFE03020001FE0003000103FFFFFF0001000201FF0301030000F90002FF0500FCFDFE01000000FEFE00FD00FDFF000003FF03FCFDFAFF00020600FF00070201000000FDFF00FF0400FE0500010100FD00FFFFF800010304FD0000FF0000FD0304000201FF00FE04000500FF0103FDFC03FCFF06FE02FFFE0003FF0202020202FAFEFF0000FCFF01FFFC0002050302FF010002FFFDFD00FCF802000000F800F8FEFD0300FD000004FD02FBFEFA0300000000030002FB04FEFCFC020001070400FDFAFCFE000605FFFEFAFC00FF00FF02000304FA000503FD01FFFF00FDFBFBFE020003FFFF0001FEFE00FD03FFFFFFFEFB0300F90001FD03FC040206FF0002FC00FE01FC00F80003010404000101FF0203FF05FFFEFE0300FEFC03000000FB00000006FDFBFF0401FCFEFF030500FF0202FD0100FF030101000000FC020004FFFB0300FAFD02040000FF00FEFF04FEFD00010000000002000100FE000300FF00FDFD01FDFE030003030200000305FC00FF0500FE00FC00FE020003FC0308FAFF00FBFE02FCFDFDFEFCFE00FD04000001FE000001FDFE00FEFF0103FE0003FD00FF03000006FE000202FFFF0101FFFF03FEFC0301FE0300FEFCFFFEFFFD000301FDFD0104FCFF020500FD02FD04FD0300000303FF0101FFFE03FE0101FFFDFE010100020001FE0002FE00FA00FE0304000000000203FFFDFFFAFE020102FCFFFDFC0102FEFC0203030201FF0000FF0002000400FF0102FD05FC01FFFF08000002000500FE00010001040200000102FD0100FB0201FFFF00010800FE00FE03FF00060101FFFFFD0002010002FF00010002FE02000203FD03FE01FF0201070100FFFFFE0001FF00FFFE0100FF000003FFFDFF010000020400FE0201FFFFFC0000FE020002FFFE00000403FFFFFDFE00FF0000000006FB0002FEFC000401FFFD000101FE0202FF0100030000FE0202050101050000FC000001FE0000FA00FE01000102FF01000102FE03030000FE0201FF000406FFFF00FFFE0001FE0300FD00FF0000FD0108FFFE000304FFFD0001FFFBFF010401FF0000FE0004FEFC00FC02FF030400FD0101FE0001040003FE0201FB000200000004FC01FE0300FFF90003FE0000FF05000004FBFC02FD000100FFFF0400FF0000FC02FE02FF0200040002FFFE010000FF01FE07FBFD0200FEFCFD020001010002FF0001FEFBFF0406050302FFFE0301FDFDFC0301000204FDFF030003FC040100FEFC04FE04F802FEFC0301FE02FF01FFFE0300FF0103FF030001FC01FCFCFCFE07FD05FB0002FD010204010003000201FE00FE0202050200FE000102F90100000002FB0004050502030003FE0200030000FB00FDFF0201FE0003020000FFFFFFFF02FB04FD030500FFF9FD0000FBFAFCFC00FB0506000302000000FCFCFF0103FD00F80200FEFF0105FC020000FA0000FB0000FF01FD010200010102FE02FB00FE000302FBF90005FFFFFFFDFE010200010004FFFF0100FCFF0001070602FDFC0004FE00030207FFF801FFFF02FFF9030201FD01010100FF010200FF0103FEFFFDFF01FFFF03020005030101030101FC0100F80603030100FE0000FEFD0000FB00FE0100FE00FEFF00FD00FFFFFE0000FDFB0300030104FF01FB030207FB01020500FD020505FEFF000007FE0001FEFB03FE01FDFF020104FC0604FD030000000000FE02FF05FEFBFBFF0200FD00FEFE00FF00FF020002000202000100FBFD0204FB050103000000FFFE0700FFFE000401FF02FEFA00FE04030204FF0000FF000002FBFE0100FE000002FD02FF0004FFFE010007040104090202000500F902FA03FA010004FF06FE0103030000FFFC0003FF0602FFFF00FF01000203FE0204FE00FDFFFE02FFFF03FC000404030000FF020103000307FF020001000000FFFE000005050005FE00FB0000000001FB030000FDFB00FEFE00FC01FBFFFCFF03000200FE0100FEFC04FE00FEFBFEFC010500FCFE0402FB010000000300FE000002000002010400FC050201FBFEFB00FE0200FE0002FC0000FDFB0201FFFF050102000000010001FE00FCFDFD00FC01FE020100030000FEFD0100FF0003F8FFFF01040500FEFBFF02FF01FD0304FEFBFE0200FF000502FE01FE010002020102000100FDFCFF0200010000FF01F90205FF01FF08000101FB02FE01FF0002000102FCFCFE0500030000FD010203FEFC02FD00FF04FD00FD0403FF02F9FFFD02050400FB0200FE02FD0103FFFE04FC0206FB000704FE010202020200FC00FDFF0504000000FFFE03FE0200FE000100FE0200FF00FE00FD05FB02FEFF04FBFDFDFEFE0602FD02020003FFFDFF01FF0003FD0300FD00020001000001000500FDFDFBFBFFFBFE0000FDFF000300FCFF01FE0100020105FAFEFD0200FF03F90000FCFE00FD00040600FF0003FF05FF000001FEFE00FCFE0402FC02000303010000FF00FE01FE030400FF0102FE04FC0000FCFD030001FB0401020201FCFCFDFF00FEFF0002FEFC020002000401000304010400000002FE0300FD0200070100FB0103FEFE0100000103FDFC00000501050000FFFF00FD070202FE02000100FE010202010000000202010001FFFBFD04000004FEFEFC02FD0000FF0001FD0000FF0101FEFCFC00FC0102020200000003FFFE03F90401FD04FFFD04020001020501FFFD0000000000FE0105FEFE000000FF0100FEFCFCFF000302FC0100FD0002FE00040002F900FF0200FB020000000200FE010704FF00FF00FE04FFFFFC00FD0102FEFF0407FE0000FF00FDFAFFFE0000010001FFFF000007FFFD020102FD000200020000FE0001FC0100010400040104FF030306FB0101FF0500FEFC010002FBFD00FF01000003020204FF01FB02FF000301FE03FDFF0102010501FEFF03FE02FB0000FD0000020003FAFCFCF8FDFC000000FC0005FFFE01FB01FF0007FF0AFE01FC01FE0102020100010000FE00FFFF02FDF9FEFCFDFE000400FEFE0300FD0500FC000100030200FD02FDFEFD0000FDFBFEFD0000000000FCFD00FC050302FD01010001FCFE020200FEFD0003FCFF05FCFC08FF0300FF00FF010000FF03FF01FB00FE0000000100030100FF0200FC05FF03FFFE01FF0400FDFCFF01030002FDF9FBFF00000300FD0101010005000206FC01FFFFFD050300000303FFFEFD0403030001010000FEFD04000003FD0001FE050000FD0002FE0001030206020002FCFF000201FA00FDFEFEFC000300FD0000000100FE01FDFCFDFBFFFC010200FFFFFCFE01FD0200FFFE0002FF000001FE000004FEFC00070003050400FF02FF0003FEFF000000FCFFFE000107FB030004FD0202000006FE000100FBFF010300FFFDFF070001FA00000500FFFCFD0000010100FF0100FD02FF01FF000000FAFFFFFFFEFFFD0300FD0100060301FFFF01FF0101FF0000FE0203FCF800000201FDFF00FB04FFFF000300FE02000001010001FEFCFFFF000101FEFAFEFEFF01FDFE02FA04FC0000FFFEFE03010304FD000200FE00FC0000FAFFFEFA000201FEFE01FFFCFEFF00FFFD02FC030203FC0005FE00020102010100FFFE0300FEFF0203FE05FEFF000002FEFFFEFE0009FF0004FE01FE02FFFF0000000000FEFFFEFF01FCFF0004010600FEFD0200040000010001FF01FEFC00FC0101FFFEFD04FD00000003FA020000FAFD0000FEFF030202FEFDFC00070501FAFBFF01000000FF00FA050205FD00FFFFFE0204000001000302FF00FAFEFCFF040001020002FF0006FC020301000004FF02FD00050002F9FDFD04FAFE0100FDFFFF0501FEFE040100FD0004FFFE0101FEFEFD01040103050200FF000406FD0002FE01000206FC00FE04FB000101FEFF0000FEFE03FE030400FF0000FEFF0102FDFBFFFBFC01FE02FEFF0003FF02FEFA000303030301FFFC01FE00010000000300FDFFFEFE03FEFFFE00000101FEFA000300FF020100010002FE000401FD020200FF00FE00FFFF000200FD01FB030201FBFCFA0301FE0003FFFBFEFEFF02000303FE01FE01FF00FFFD07FD0501000301FF02020000FF000700020100FF0203FF01FD00030301010004FC00000302FD00000401FE05020100FFFCFD000003040500010000FC07FF00040100FDFB010200FD03FEFF030103040602FF0002FFFF0003FE0100000100FE03FD0000FF00FA01FF000104040000FF02050005FEFFFF00FE000102000101FF03020202FC00FEFCFF000002000000FE0200FFFFFF0502FEFD03FF0501030000030001FC03FA000100FB000700000002000102FEFCFEFE0000040504FF01FC00FF080006FFFDFCFB00000000FDFF03FDFB00FC000102FFFEFFFD00000201FEFC030101FFFC00FDFC0201FE0000FFFD0101020001FE030000FE0003000002000002000600010100FFFE01010402FFFE02FFFE020201FF0000FC0203FE00FAFCFE030200FE0300FFFE0000000300020501FF0400020000FDFB0103060205FE030001020000FCFE02FEFF050406FBFFFFFD0205FE00FAFE0101000503FE01FEFC0000FD01000000FCFE01010203000200020300FF060001FC000100000100060002000001FDFF01FC0200050301FF00000108000000FDFEFEFFFDFC0301030300FFFDFF010003050000000000FB000300030101FFFF010100FD0104050002FDFA000100FF0002FFFCFEFE0201FEFD0002FFFCFF00FB00FCFE0000FF0100FEFF06020200030302FDFB02030100FFFD01FC00FEFA000600FAFE02FEFE010004FEFF040100FC0400FCFF0000020002FB00FD00010104FF02060000FE0203FD04FDFC00040000FDFD00000307FEFF00FD010000FC0200000104FCFB0601FEFD0404FD000504030300FDFD00000701FF0200FF02FDFE01FB02FDFF010201FFFDFE01FD01FC02FDFCFFFF02FAFE000201FD0306FC010003000000FE01FBFE030000040002FEFF0105000008000100FA03000302FB040301FF0201FEFFFDFFFB010400000302FCFFFF04FEFCFF050000FF00FC00FFFEFDFB05FAFD0001FE010102FFFE0407FE02FA01FF00010201FF0100000000FE0200FC0000FD0001FCFEFF010000FE0102FF02FE01FFFBFFFE0001000300020003FB0405000102FDFDFD0201FCFE0301FFFFFFFB04FE00000000FF0700FBFD00010500FE0100FF05F80004010403FF00FFFC0002FF01FB020200FDFCFE01FF0102FCFBFDFD00FC00FB01010203FF00FFFDFE00FFF9FF000003FF060301FE0302FDFC0401FDFD040007FD00FC00020003FDFEFEFEF8FB0000000101FC00FD000004FEFEFEFF000402FEFB010100FD03FD04FF03FB0000000001FFFD00FF00FEFEFD02FE010004FE00FEFFFC03FD0202FEFDFFFEFF0406F901FEFE010203FE03FFFF00FEFD0000FFFE00FD00000005000001030000FFFEFFFFFF020301000203FF00FEFBFE00FEFE00FBFF01FF04FA0000FE05FBFD0000FDFE00010000FFFD02FD0206FEFE010305FEFF0300FD0202FFFE000201FE00FD04FD030000FEFDFEFA0000020101FC02FE00FF000002FEFC0202FEFEF9FD03FEFD02F8FFFC0401000000000005FBFD00000006F8FC0100FBFDFC00000606FEFEFFFDFFFFFE0000FEFB00FF01FF0206FFFBFF03FD00FFFF040205FF0302FF0100FDFFFD00000000FE02FF0000FF03FE000300FFFCFE00FD040000FE01FDFF01FE000002FC0401050002000105FF010100FF03FEFCFEFFFDFFFC0001FCFE040403FA0300020001FD00FB01FDFC00FDFC0000050000FF0903FD0200FD02FDFF0101020200000001FF0000FDFE0001010002FD00FF01FB0001FEFC00FF0304FFFFF9FE0202FE0002FE00FC00FD01FC01FC00FFFFFD04FF03FD00FFFFFD0400FF01FFFF00FD00FF02FF03FE000502010000FB00FD020201040007FF00FEFB00FEFC02FE04FA07060002FEFD030007FEFFFD02FF020301FC02000505FBFEFDFEFFFFFC0005FCF7FF01FD0200FFFD0100FA00050000FA00FF000002FFFF02030003FDFD0004FA040502060001FF050200FE0003FF00F602FF05FFFB0200FFFF02000001FF03FD06FE04FE0000FB0200000A02040200FF00FBFE01FFFD00FFFB010301FCFDFEFA0003FC03FEFFFB000400FDFB030101F9FDFD01FC02FE0000FEFF03FF020400FF05FC02000200FF0003000601FBFCFBFD000003FEFE00FD05FD0002000603FBF8FF0202FE000002FE040303FE0002FF010201000101FE00FE00FEFE00FC00FE04000004FEFF02FE02FD000103FD010302FC00FBFEFF0201030100FEFD0000FE0104000001FC000103FFFEFB0200000301FC00040000FF000100FD00FD03FB01030103FF000101FDFE00FF02FEFD050003FD00FC0204FE00FF050303F6050003FFFDFBFD060002FD00FCFC0000040302010000FEFFFF0102FD03FDFE040004FF04000301F9FEFFFDFC00040402FF060001FDFF0004FC0001FD00FD0702FFFE06FB01FFFF030100070002FE03FE0007FEFFFDFE000004FCFD0000FEFE0200050003FFFCFF07FFFD020300FF0300FD000002FF0200FB01FDFF0100FF04FE01000100FF00FEFEFEFF0200FF0402FCFF050500010000FD00FF00FC05FD0303FC0200FF01FD02FF030401000205FF00FB01FEFE00020005030303FFFF01FEFF00FD0301FE02000000000000FF0204FFFDFF000503FF000500FCFD0000FFFF0403FDFB04FDFFFE01FF010106FEFF03FEFEFE00FF00FE00010001000004FDFFFE0000FE03FFFF0400FC01FFFF0102020403FD01FE010202FBFD0100FB03000004FD00FEFF0504FF02010000FF02FD0302FEFC000300FD010000FBF70100FF01FF0001FC000100FD0001FD0202FE000201010100FB03FFFE06FFFF030201FD00FFFD0002FD040003FDFFFF00010004FD0000FE020000FFFF00040100000101FF050004FE00040000FF0403010003FDFF05FD01FFFE01000003020300FCFD03FF00FEFFFE0503FF01000003FB000001000303FDFE02FFFD000000FD01020001FD040300FFFE00020202030001FFFEFE0304FC0002FB0006FE0003FF040103FD0100000200FFFFFBFE00000000000300FF05FC0000FF0201FF0100FC02010600FEFD01FF0209010001FFFEFEFF0003030401000302FD02040102FFFC00FCFBFF0905FE05FDFD010105FAFD07FD0805030000FBFBFFFE0006FFFEFEFE00FFFC0000FC0002FF0201FD0202FC000100FD00FE05FB00F900FD0000FC0100FEFD0401FF02FF0000FF03FD05FF00FDFB0000010002FC00FDFBFF01FDFF03FB000003FF0000FD010103FBFFFC000000FD00FD010BFBFDFF03FF050100FFFF0003FB0003FB00030100FEFF00FC0101FF07FDFBFE00FE03FFFC050002FE00FCFD02FFFD010403FD000205000100020002FF0001FFFF000501FEFF000001FBFE03FFFF01FFFC00FB0000FF0003FC0100FD06FD000302FE030205FF00F900000002FD0104FD040202FC02FDFBFE000303030000FF04FF0003000201FFFC00FF050006000300FE0300FA0000000000FB01FB04030003FFFC010001FFFFFE02FE01FEFD000104FEFC00FFFF040400FF00040600FBFFFDFC00060001FEFF00000400FF00FF02FDFF020503070200FA00FEFFFF00000003FE020001FE02FF01FFFCFE0100FB03FE0104FB00FF00FC00FF0003FE00FDFC0102000001FF01FE01FE0200FD00FF020100FEFE02F9FE00FB0204FD00FD0001FFFF04FE010600FEFF050103FF0001050603050000050200FEFF0006FDFFFEFC0005FC00FAFC02FF00FE030000FCFF01FF01FF00FDFFFE000100FD0100000000FFFFFE02040000FF02FEFDFD00FB010001FE01FD03FDFEFFFD0002FCFB0401030002FD04FF0200FEFE00040100FD0104FE0201FF00FD01030400000300FE00FD040201010000000203FE0100FDFFFE050003FEFFFF000506FD0000FC0101FF01FD02000000010000F9FE08FF00010003FEFDFCF9FD00FB00000100000600060103020202000106000200FD00FE00FC000002FE0205FB00FEFE000303FF0102FFFB0107FF00FFFE0201FC00000202FB000004FD00FD00FDFB0001FA00FFFEFDFC00010000FC010001010003000100030205FCFF01FF0003FFFFFF01030202010003F800FAFD02FA010400FE0000FC02000204F7FDFF01FEFFFFFEF8FDFC010104F9FEFE"> : tensor<100x100xi8>
    return %c : tensor<100x100xi8>
  }
  func.func private @expected() -> (tensor<100x100xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xEEEEADDFEBFDEAEFCEBC8EFE377BF3B7F3455BBEFFFE7EFB9FBE75F9EEE767FFFF5FEF678EF77FEE9FE95DEDDBD9EFF3FAFBC7D7FDB797AFF5F8AB8AAFA6FFFFF3FF7FF9F6EF7FBE7FFFBFBE79D18F3F6EF7BBFEB97FBFF5DE634FDDE7BF7FF7E6FFF7FEDFF6FFFEF8CDFE7CCF7EAF7BEFE7FFEF37FB3E7ECDDBFB6EBF372EAAFFF3BD7B73FF7BB353FDCADEFFFF4F1FFABEFFDEFF7F972AEEB7C1EF7ED0E3FBB4DDAD1AFF9CB2FFEEFDFF3A3ED6EBFB5FFAA3CFE7AF6E722FE3BFF119FFF6FCB7BE8FD52DFFF3A572FAFFFDFAB9557BFEDEF5EFC52C2AFBFDFB7FBE5BB9ADBCF97B119E2AFDBFB3FF73BFFFBCA3BBB5FAEB87EDF49A6FEFEFF9FFF17A7FFFA77EF3D9FEDFFBFAFFB7FFDCF43EBD5E6FDD16FF5A73BFFFFDBCFBAEFFDFDEDFBE7CDF7FDD3FF7F81CDCA5D6E9E72FF3BF3FF9FFD67BEAFFDD9F6DE76BFDF77F6FD7FDF5517F7FF8FFF9E3F7EC3FFFEB4FE97EFEEEBFFF35FF8CF5BE7DEFBFCB97FEF9F7EF70DBDBAFFFDD57747FFFCDFDF3EF2BDD5DDFFAFFDDCDFD3FDDBB49F5F616E9E57D6FFFD3FD7ED31ADEFF7DF53FFD79FF6CEFFDDEBFBFBF6753EEBDCB356CF6E4FFFFDA8DA47FEBEFF5CE01E919DE4CF6F5F3A7DA2BAFAFFEDBABD6D7D7FEE8FEBFEF5F6F1FF6FB5BFBF6B7EEFFFF7BAFF07EBDCFFFB7FBCF73CBEF5FF4EF7ECEDBDFB0DF9FFDABF26E5B5D7AEEEFDDFCFF6FF7D7AFBE7FFFE3F7F5EB73B3EFDF7DDFDFE7E2EE39BFDFE177C3B587BFF5F5D33DD6FCAD63DBBA2777EBFFFB9FEFF78E5EFED0DFBD5FBBDF77F7AF7FBBDEFF1DF17FDF63EFE90B55FDCEADF6FB6FE6BECDFEFF7EFFFBF9FFD7D6E1FFFF9FEA7FAACEFBD6FDB6EEFF6E97CF7E767E61F7FEF227D3DEE7BD5DF9DE3FDDF7F6C5B799AFCBAFE7EF6DFFFFFBDEFFFFBEFFFB755F8F7D97BEF9DF79377EFB49BE5FEF5F6FBFFDFFEFFEDF7F5FA6B5CEFD7F3FFBFF07FFB5D6FAC7EDF79397F7FEFAFDD96FBF7FE7BD982D9CEDAFBDDF8F92FB5BF31FEDBE5C7FFF5FBFD7CB7F7FFDF2EFFE6FFFEF5F8FAF56FFFFFDAE527FD6F777D67AEE77FA7EEEFFF6DAE3B69FE7ECD7C77D7E9ADFC7FFF7838FBDD7DA465FDF3F3D9DBFD2EDFFDDCD7FFFFF497FF4F7FFD27FDD6A9F07FBFBF65FF345DB7FDFA7D77FFECDD969DFDD2BFADFDF66AEBBE3DEB3DE9BCEF6F16FFFF379AFD9F7FFF3AFF2FEFBB7EFFEF9F6077F77E9FBC7997F7FF47DBA6FFBBCFEFDD7FFAFBBBBE797FFDFF75FF4CFEBDA57EBFFFDFDFD3EB5DF9BF33B97DBFBBF3B9DEEDC6BFF4EECFFE955F4F3DB87AF9FFF3EDF29A9FDFF79DF465FFE9BFFBFF7337E6D4BE5F78CFFBD41FDBDABFF5EB3EFDF5BBF6FD3FAF9CBBC99FF7BF3FEFFFFEF2F3E7D6AFFFFFC4FFDFDFF3E2C7DFEAF7AFFFF61DBEEDBEFFFFADDF9FF5BFD71F97E7FB87EFBFFFF6F16997F77FB7996FFDF77CFF3CBFFFFE078E73FFBDFFF0EE7BAF9F9FAFEFBD7DEE4FBC76CBBFBDFFAFEDEEB3B7DDBBFF77FFFFD5D29DFFD5EFB3BFFCCEBFBEFD7FFF37BD5CF5DFFFCBE6BCDDFDFDF9CFD5CAA7FEFDBFBFD36CFFFEEEFBBEDFFF53EE7FA6FB9EFD5F797FAFB77AFFF3EF0BBE6FFFFABF27CFFFF9BFDF96BEE56DFFBBFDD6B66F396F79F7B7FA63FEEEDFD6D7F7AD0F9DEDFF7FEFBFF3FEF9BFED595FDF9A6FD337FF4BF6DDBBF6FFFAD76CF7FE9EE2FFEBF73BDD582EFFDF75E7B9DFAAFE3FEFB557BFEBAEFF5F7FB6CDFDEFCAEE1F3EDBFDFB7FD3DF5F1DE3B7FE4F54BFAADCAB7FF7DCEDA5EB6FAFB6FBFECFFFF"> : tensor<100x100xi1>
    return %c : tensor<100x100xi1>
  }
}
