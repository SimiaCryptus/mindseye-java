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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class MaxDropoutNoiseLayerTest extends ActivationLayerTestBase {
  public MaxDropoutNoiseLayerTest() {
    super(new MaxDropoutNoiseLayer(2, 2, 1));
  }

  public static @SuppressWarnings("unused")
  MaxDropoutNoiseLayerTest[] addRefs(MaxDropoutNoiseLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxDropoutNoiseLayerTest::addRef)
        .toArray((x) -> new MaxDropoutNoiseLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  MaxDropoutNoiseLayerTest[][] addRefs(MaxDropoutNoiseLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxDropoutNoiseLayerTest::addRefs)
        .toArray((x) -> new MaxDropoutNoiseLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MaxDropoutNoiseLayerTest addRef() {
    return (MaxDropoutNoiseLayerTest) super.addRef();
  }

  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends MaxDropoutNoiseLayerTest {

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

}
