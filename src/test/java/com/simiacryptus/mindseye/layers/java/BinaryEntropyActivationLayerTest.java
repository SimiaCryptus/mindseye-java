/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;

import java.util.Arrays;

public abstract @RefAware
class BinaryEntropyActivationLayerTest
    extends ActivationLayerTestBase {
  public BinaryEntropyActivationLayerTest() {
    super(new BinaryEntropyActivationLayer());
  }

  public static @SuppressWarnings("unused")
  BinaryEntropyActivationLayerTest[] addRefs(
      BinaryEntropyActivationLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BinaryEntropyActivationLayerTest::addRef)
        .toArray((x) -> new BinaryEntropyActivationLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  BinaryEntropyActivationLayerTest[][] addRefs(
      BinaryEntropyActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BinaryEntropyActivationLayerTest::addRefs)
        .toArray((x) -> new BinaryEntropyActivationLayerTest[x][]);
  }

  @Override
  public double random() {
    return 0.1 * Math.random() + 1.0;
  }

  @Override
  public RefDoubleStream scan() {
    return RefIntStream.range(50, 450).mapToDouble(x -> x * 1.0 / 500.0);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  BinaryEntropyActivationLayerTest addRef() {
    return (BinaryEntropyActivationLayerTest) super.addRef();
  }

  //  /**
  //   * Basic Test
  //   */
  //  public static class Basic extends BinaryEntropyActivationLayerTest {
  //  }

}
