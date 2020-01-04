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
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class CrossDifferenceLayerTest extends LayerTestBase {

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 4 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new CrossDifferenceLayer();
  }

  public static @com.simiacryptus.ref.lang.RefAware class Basic extends CrossDifferenceLayerTest {

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") CrossDifferenceLayerTest addRef() {
    return (CrossDifferenceLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") CrossDifferenceLayerTest[] addRefs(CrossDifferenceLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CrossDifferenceLayerTest::addRef)
        .toArray((x) -> new CrossDifferenceLayerTest[x]);
  }

  public static @SuppressWarnings("unused") CrossDifferenceLayerTest[][] addRefs(CrossDifferenceLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CrossDifferenceLayerTest::addRefs)
        .toArray((x) -> new CrossDifferenceLayerTest[x][]);
  }

}
