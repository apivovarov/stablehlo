// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.log_plus_one %0 : tensor<20x20xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    return %2 : tensor<20x20xf16>
  }
  func.func private @inputs() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xBEC5C8B98642ADB7D2BB33C058B4754014BCC540A8BEFA38BF4271BF7C459EC46EBDF4C0663E844322BCE54098C46DC27ABEA9BC62C32ABCF7C4CFC596C0D8C4013ECB3CC5BC9038A43DBE4264BB2FC6F1BC004041C4E23CC4BC4FB9D1C58D3C653404470D337BB7633749BC17C523C0613D48BCB8BD0039BAC4784191C3B6388BBBF844E7C469C4E7BCCDC195C40EBBB734523E09BCC2C6BB435FB8A6C168BC08BDC5BD9EC414BD77C1ABBA08C597C1C9C459B732BF8F3D9A38EE3D2BC163C0A530E340B3B68C36B2424FC459C2D6C09DAFEC4190C0A7419E3997C2713C1FBFD33847C4FCC4A5BD23C1B7BF374819C1FD4339C7EFB906404AB847BE193B354118401142F2470D41C64184C1383D65B9B23660C2033DE847AD3442C3D63C6AB83848FB39873521440741F4C5C1B930B9C7A48DBDE143DB43BA40AAC28639CF44E9BE7D438B2D9EBEDE3F1E3E8F40FC3B6CAEC0B4544058C45BC742BAB8C06DBEBEC0B929E8C32FB3BCADD8C17EC2EDBC43C1C438C7C24340B0C1024290B69DC0B73EC1C42BB7214863C0BC40D34047A9BDC4EFACE745AD409CB8C03D01BD47BBE1B823BD0AC3F8C6BBC4A7C5ADBE5E3C86B572349AB89AB0ACC1664211403A459FC08D48DDBE0D3C17483E3E52C53BC099C014AD10C75136B2484F41C641D8BD31C190B79DC003C3B7C3853DDFB7C2C3F53A40BE16C57F3C85BF3BBF4CC1934591B8CAAC09C06D3B92C3AD32E9BC8F4760C4A43D7B3D6741404038BD5538143002C500BDC9C15AC229BE28340941E2B7C64012C84E3C48453AC426BC1944DEC389C4C740C740D04488BB3D453948CFC5A63D7341594477402DC3FAC29AC023C429BF6D3A2B4523BC2BBBE8C4AD3630BE7B4569C4734406B97E38744578C446C440BBE545154578423FBE8A42DBC12645E03DE5B971C12D4499B029C23B3D76C0DC408C400D3BAF41A8399644E7BBF63F4C441DB8113C4B32B345FBB146B93ABF2EC699C7D9BF3EC861B81CC0BABF75C00AB4B2C3DBC5ACB9CB43C8BE0940D041AABF2841A93F0ABF7740DCB5093FC3375744963B8AC030438A3CF9BF404079C052409741F5BC41B490C07EB8A0372C3BE8B80DBF99C09B34FF41D6C0FA416042"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
  func.func private @expected() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x007E21BDCC3D39B998C3007E12B5B03C007EE13C007EBD37E73D007E7A3F007E007E007EA53B3D3E007EF33C007E007E007E007E007E007E007E007E007E007E553B4E3A007E38370A3BE63D26C1007E007E653C007E623A007E5CBC007E133AC4332A405F320CB91336007E007E007ED03A007E007EC537007E453D007E6937BAC1253F007E007E007E007E007E45C02234953B007E007E543E53BA007E007E007E007E007E007E007E2CBF007E007E007EEAB8007EF93A4537463B007E007E5630F23C58B87D35E13D007E007E007E0EB0823D007E5E3D4138007EF939007E8E37007E007E007E007E007E7D40007E6E3E007E6ABD693C25BA007E1439213D753C943D62400A3D6E3D007EAE3A7DBC9635007E803A60401A34007E573A6BBA7D407838C0348A3E073D007E15BD2EBCD3A4007E633E623EDA3C007E33380A3F007E3A3E512D007E5A3C6C3BC03C8A39C4AEA3B59C3C007E007E19BE007E007E007E9929007E11B402AE007E007E007E007E7A37007E913C007E8D3D39B8007EE33B007EC2B87340007EDB3CE93C64A9007E22ADBA3FD23CDEBA213B007ECFC088BB007E007E007E007E007E007EE739C6B6D833D9BAF8B0007EBD3D703C513F007EA040007E98396F40863B007E007E007E4AAD007E5335AF402F3D6E3D007E007E1FB9007E007E007EEF3A6CB9007E0139007E007E083A007E007E007E883FC4BAFAAC007E4039007E1032007E4B40007E0A3BE83A3C3D8F3C007EEB36AD2F007E007E007E007E007E6433083D6EB9E13C007ED8395A3F007E007E843E007E007EE23CE23C0A3FADC1533F7E40007E0B3B433DB53EB13C007E007E007E007E007EB738473F007E87C0007E9435007E7A3F007EC83EE9BB2137753F007E007EBCC0B93F393FC63D007ECE3D007E443F3B3B57BD007E933EF7B0007EB13A007EEE3CBE3C0E39623D4738E23E68C4623CAB3EC6B99C39BF319C3F9FB24EBC007E007E007E007E007E57BA007E007E007EA8B4007E007EF0BC5B3E007E6B3C733D007E193D473C007EB13C4CB70F3C5536B33E5639007E193E113A007E8F3C007E9A3C563D007EF2B4007E98BA3C361F399ABB007E007E0C348B3D007E893DBA3D"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
}
