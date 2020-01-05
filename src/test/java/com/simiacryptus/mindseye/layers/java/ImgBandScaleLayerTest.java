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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class ImgBandScaleLayerTest extends LayerTestBase {

  public static @SuppressWarnings("unused")
  ImgBandScaleLayerTest[] addRefs(ImgBandScaleLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandScaleLayerTest::addRef)
        .toArray((x) -> new ImgBandScaleLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgBandScaleLayerTest[][] addRefs(ImgBandScaleLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandScaleLayerTest::addRefs)
        .toArray((x) -> new ImgBandScaleLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    ImgBandScaleLayer temp_76_0002 = new ImgBandScaleLayer(0.0, 0.0, 0.0);
    ImgBandScaleLayer temp_76_0001 = temp_76_0002.addWeights(this::random);
    if (null != temp_76_0002)
      temp_76_0002.freeRef();
    return temp_76_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgBandScaleLayerTest addRef() {
    return (ImgBandScaleLayerTest) super.addRef();
  }

  public static @RefAware
  class Basic extends ImgBandScaleLayerTest {

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
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
