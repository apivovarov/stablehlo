// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>)
    %1 = call @expected() : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<5x6x7xf16>, tensor<2x2x2xi64>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    return %2 : tensor<5x6x7xf16>
  }
  func.func private @inputs() -> (tensor<5x6x7xf16> {mhlo.layout_mode = "default"}, tensor<5x2x2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xEBC4A9BC6F4318C209B96A3AB7B4333FA63FB2C07635C6408FC1F13449C59FC4E52F31443A3BB9441B2A243F2136463FC038670C66C04CBD144619C4B54047C06F352DB6753F383CAA2D4EC5FE3AF1B3F6AAFDB21242C6C065C24ABF50472AB968C238440F3B0CC19C408AC4BFC24834D3B4C33BABC538C1EA42EEBB7C3688C4053F1445C239A03DC8438C4549C1713D0243843E774195B57336333CB135E340A7C388390342CA459C30104163410D434E4410406EBDC23952443EC30446F341B82E3A3ECFC0F2C1A1455F44D93F7DC6C734D5C0E2C03737AB3E7DC431431EBBC743BDC2C841F045BB3E67422A3FA74178C0074146C314C06ABBED4625C0E740033C923DD63E74C025C2BE424C4372BC833C8D41E4366A410435AA3D9C39A3C5CEC6BE3EF43DFD45E430F24063381F3ED1BD4F44EA3B9CBE873BCBBE76C147396A4456BD8CC058C420C10A42233C1C410CBE4544913F1343C5C3BC3B1DC16645B84176BEC5C182BC9243B938ACB9DFB0C13C7ABA0BC44D3CA7441343133E95410F46D8B684414338E6456ABCF3BFD6C0E3BD23B0E4BE75C24AB08A48B9460D440C315E41"> : tensor<5x6x7xf16>
    %cst_0 = stablehlo.constant dense<[[[4.355470e+00, 1.568360e+00], [1.973630e+00, -1.762700e-01]], [[2.361330e+00, -8.393550e-01], [-1.362300e+00, -2.796880e+00]], [[2.298830e+00, -1.464840e+00], [1.072270e+00, -1.643550e+00]], [[3.105470e+00, -1.664060e+00], [-1.618160e+00, 8.422850e-01]], [[-1.998050e+00, -2.687990e-01], [7.524410e-01, 1.403320e+00]]]> : tensor<5x2x2xf16>
    return %cst, %cst_0 : tensor<5x6x7xf16>, tensor<5x2x2xf16>
  }
  func.func private @expected() -> (tensor<5x6x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xEBC4A9BC6F4318C209B96A3AB7B4333FA63FB2C07635C6408FC1F13449C59FC4E52F463E3A3BB9441B2A243F2136463FC038670C66C04CBDE53F19C4B54047C06F352DB6753F383CAA2D4EC5FE3AF1B3F6AAFDB21242C6C065C24ABF50472AB968C238440F3B98C19C408AC4BFC24834D3B4C33BABC538C1EA42EEBB7C3688C4053F1445C239A03DC8438C4549C1713D0243843E774195B57336333CB135E340A7C388390342CA459C30994063410D434E4410406EBDC23952443EC30446F341B82E3A3ECFC0F2C1A145DCBDD93F7DC6C734D5C0E2C03737AB3E7DC431431EBB4A3CBDC2C841F045BB3E67422A3FA74178C0074146C314C06ABBED4625C0E740033C923DD63E74C025C2BE424C4372BC833C8D41E4366A410435AA3D9C39A3C5CEC6BE3EF43DFD45E430F24063381F3ED1BD4F4479BE9CBE873BCBBE76C147396A4456BD8CC058C420C10A42233C1C410CBEFEBF913F1343C5C3BC3B1DC16645B84176BEC5C182BC9243B938ACB9DFB0C13C7ABA0BC44D3CA7441343133E95410F46D8B684414338053A6ABCF3BFD6C0E3BD23B0E4BE75C24AB08A48B9460D440C315E41"> : tensor<5x6x7xf16>
    return %cst : tensor<5x6x7xf16>
  }
}
