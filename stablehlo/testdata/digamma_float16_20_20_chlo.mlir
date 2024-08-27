// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = chlo.digamma %0 : tensor<20x20xf16> -> tensor<20x20xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    return %2 : tensor<20x20xf16>
  }
  func.func private @inputs() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xAFC252C1133D23445F3ED2B991BC8EC5DF4044396AB862C296C175C23F4939AA4B3D073C9244EC357A4638B05DBE784212369CB874B977C14341E4C026C4C94027C8C33BA3BA1733ACBCBCBF43C08B40CB422FC4E7BC6E35B33C30B5D7B592BE4943F8BFCE390CC266404F38D4B7CB3C22BF95B20F40EEBEDEC51BAF8AC2A62B5D3F77C46E3967C867C04F4047BBB23F3C3B8740FA3CDE3C8444344189B96E452EC63840873840A856BB0FA7554284429AC050C43CC22ABF3EBD69BDD0BE3C3E8DB4FC364BC306BAB3B9FFBEBCC1C62EA23EAEBEEFB6AB443534A6C7C23E19C63141914621314BBC1439863D484088BC77C4AE3A1835DB45D34494B8AF41EBC38EC24544AA30833A87408FB3C02F0FBC3FC8514089C10D3E1BC21FBC8141E740C3361DBEDABB5445E2C392BFA2C4A6434E3E5CC23841B7431F383C424DBD03BECE4498301745A43D8EBEA6BC27C71CBCB3410845E1C1C3414F4419BFBABC03BC31C15C3EBAC3A437AD3C56C0A7BC1A413D437344BF3C0EBC6A326E40E728BA37CAC0903B4FB37B42493C6142F645D14312447FC2AEAF57AC2AAD5935074486C1ACBAAE460C41CCBFE1C13B3A98ADD3408EBBEAC2E3C1C2448F3806404C44614019BF53384FAD2040FB350EBA56BC2D4654C48B464EB7824061BF98B8943A54C523B91EB9613E80413AC6FEB915C0C0333FBE1A40D5BC404014431EC19741333BF8C1203D05C0D5BD2E4458C05A3C3C4504C0CF40E8A3123832BDFFC1133F443F853AEF45029E45C434323444C23AED2CF240A9C0F7B9F3B491C593392EC0772D20B9024504BFFC3D58443A403E3DC530C4C35C44E746EDC10A3AB0BDE4BEE3C083B8543D2BC301C286BDBE2A73431142343E212DDE38033DE93C81BDCA43D73E1542A0420F2B6341FF3E64C6594386BCB132FB3F9AC2C0BA0D3E97C3D73CA647D53F2D2E93376941BAC11147CFC0F0C655B0C03C89BE6F3D6AAD81B816B71441E4C6B5C14FB473B97CBAF23694BECCB505B6D942CD43A0BCA840F63CFB3CE23447B55AC3374136B98FAA95407840373A7C3F7640A9425D30A6419FC00F41F93B38C36F391645D7B7F4C11845A543CEBFE8BE343975C1023277C23745E2C01B40"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
  func.func private @expected() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x1B4298B893B22D3DAB2FFEC01A47073D5E3961BDE8B6E345DEC1EB449B40FA4C95B087B8A03D97C1283FC44698B10C3C70C14AB94ABF8BBE203A853EC8473139634405B9FDC4C7C4EB45F9CA1748AD38463C6046394425C225B52D3F483DE9B9993CF1D799BC7B515A3824BF3D30A9B416C2C443103701C03CC51E481C444ECC0935CF3F21BD76C04A45243834C9F635F9B9A43881B34AB4923D043AD7BF633EDC46DB37AFBE5F4FB4C97550E43B143C9D425843B7487BC27741654021BEC12DAE40A1C03EB0E7C183C09CC0DFC5EFC898314BBC1E39B83D04C423386832E749FE39373F94C6F14AAEBD32AD14389847CF3F19BB94C2B83EDE3DFEB8DE3AB0CDF343523D39C777BBA4385D4253C84A548C402938A5C02E2A044D2C508F3A6E39CFC0FD36AED24E3EB1CB09C8D433D23CBF2E44460B3ADC3C90BFBF3B13416839D93D54C71A3E88AA8CB926460C489D50E53A0D3E8ACBFF3A5C3DACC16E45575D2031822F62C528C044B561461C46D239913C813DE6B4985447C56C38A8CE1AC04C405EB9B1420E3C73B7F63BCC3EEB3C1A3D81448A47074BD6493FC20E3D64C028C54A3FB739A8CC8ACB0FBC5B49463959CC383F0FCCCE3D9FBEE336593D4E38ACC11BBFAB49663787C10EC20D4AF33EEB42333FD436993810C524B952BB4B435EBD42BDC72F8D3A8D45C0C14C4E5EC48D2F4837A7440138773C4D38B53A0ABAD6D31AB27556663C393D3B460FB73A3E07583E390254AFBFCD41FBDF2F34C13473BBC73E4F59584475C5403DEEBABACA8439CE41A0C1F23F8E3CEBBCE54914CA4DBD083ECCC03428663DE53709B111C7AAC66A3D6E3F65CE4BBCCE3D59BF993E5DB845B07D380360823FDFCCB33C7D3B4F2D77CA0BBE2BB313B4B93FE73CED32833B283CA8CC5A3AE5331C42A33CB6470FC5B736384390C52E2A72C16CB4DF3F533665C934C0643AA4C5883F1740E5CA9046E1B419B9B7AE8C494AB86A38C739C7C620C5284144BF59C4A9C018BA663DCE3C503CE93C6546EC38A8B377B3E1C2E63E08B8093ACBBDB74CC338833814BC61357E382E3CB5C7CF3A5542BD39AAB8AB341FBD1A3E08302BD11B3ED13CDBCC9BBF7ABD45BEA2C5D444363EAD3E4D37"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
}
