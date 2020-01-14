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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class RescaledSubnetLayerTest extends LayerTestBase {

  @Nullable
  public static @SuppressWarnings("unused")
  RescaledSubnetLayerTest[] addRefs(@Nullable RescaledSubnetLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayerTest::addRef)
        .toArray((x) -> new RescaledSubnetLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  RescaledSubnetLayerTest[][] addRefs(@Nullable RescaledSubnetLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayerTest::addRefs)
        .toArray((x) -> new RescaledSubnetLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{6, 6, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    SigmoidActivationLayer subnetwork = new SigmoidActivationLayer();
    RescaledSubnetLayer temp_60_0001 = new RescaledSubnetLayer(2, subnetwork.addRef());
    subnetwork.freeRef();
    return temp_60_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  RescaledSubnetLayerTest addRef() {
    return (RescaledSubnetLayerTest) super.addRef();
  }

  public static class Basic extends RescaledSubnetLayerTest {

    @Nullable
    public static @SuppressWarnings("unused")
    Basic[] addRefs(@Nullable Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

}
