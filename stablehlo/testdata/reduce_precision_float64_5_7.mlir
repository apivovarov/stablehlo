// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xf64>
    %1 = call @expected() : () -> tensor<5x7xf64>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<5x7xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x7xf64>, tensor<5x7xf64>) -> ()
    return %2 : tensor<5x7xf64>
  }
  func.func private @inputs() -> (tensor<5x7xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.5999635273281747, 1.4827654460864044, 1.9829078643444717, -3.2814629726532631, 3.6223144819078295, -2.2519937179350107, -4.0122126839426819], [0.37984850888286292, -1.1601438971693097, 2.7871036428784022, -6.8297057470110376, -0.079031342660781017, -3.5858063300253527, -2.2257663137341401], [-1.3284106315604862, -5.8103300235365953, 3.9695748694054496, -1.5131989535057386, -1.4531285089600923, 0.99380300971033408, -3.2517037076706288], [-3.9708125542102324, -3.6015744291922873, 5.0838803766070821, 1.6167500164057285, 2.0957178509626551, 1.2844228818292567, -3.5269493009539539], [-0.69235475301926341, -9.6916697007882355, -1.6590481553827261, -2.2678685538023631, -0.17615739910966147, 4.5381716875336782, 2.8904453616748444]]> : tensor<5x7xf64>
    return %cst : tensor<5x7xf64>
  }
  func.func private @expected() -> (tensor<5x7xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.5999635273281747, 1.4827654460864044, 1.9829078643444717, -3.2814629726532631, 3.6223144819078295, -2.2519937179350107, -4.0122126839426819], [0.37984850888286292, -1.1601438971693097, 2.7871036428784022, -6.8297057470110376, -0.079031342660781017, -3.5858063300253527, -2.2257663137341401], [-1.3284106315604862, -5.8103300235365953, 3.9695748694054496, -1.5131989535057386, -1.4531285089600923, 0.99380300971033408, -3.2517037076706288], [-3.9708125542102324, -3.6015744291922873, 5.0838803766070821, 1.6167500164057285, 2.0957178509626551, 1.2844228818292567, -3.5269493009539539], [-0.69235475301926341, -9.6916697007882355, -1.6590481553827261, -2.2678685538023631, -0.17615739910966147, 4.5381716875336782, 2.8904453616748444]]> : tensor<5x7xf64>
    return %cst : tensor<5x7xf64>
  }
}
