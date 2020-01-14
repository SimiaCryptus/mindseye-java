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

public abstract class ImgBandBiasLayerTest extends LayerTestBase {

  @Nullable
  public static @SuppressWarnings("unused")
  ImgBandBiasLayerTest[] addRefs(@Nullable ImgBandBiasLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayerTest::addRef)
        .toArray((x) -> new ImgBandBiasLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgBandBiasLayerTest[][] addRefs(@Nullable ImgBandBiasLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayerTest::addRefs)
        .toArray((x) -> new ImgBandBiasLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    ImgBandBiasLayer temp_68_0002 = new ImgBandBiasLayer(3);
    ImgBandBiasLayer temp_68_0001 = temp_68_0002.addWeights(this::random);
    temp_68_0002.freeRef();
    return temp_68_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandBiasLayerTest addRef() {
    return (ImgBandBiasLayerTest) super.addRef();
  }

  public static class Basic extends ImgBandBiasLayerTest {

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
