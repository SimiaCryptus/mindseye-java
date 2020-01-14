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

public abstract class AssertDimensionsLayerTest extends LayerTestBase {

  @Nullable
  public static @SuppressWarnings("unused")
  AssertDimensionsLayerTest[] addRefs(@Nullable AssertDimensionsLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AssertDimensionsLayerTest::addRef)
        .toArray((x) -> new AssertDimensionsLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  AssertDimensionsLayerTest[][] addRefs(@Nullable AssertDimensionsLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AssertDimensionsLayerTest::addRefs)
        .toArray((x) -> new AssertDimensionsLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new AssertDimensionsLayer(2, 2);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AssertDimensionsLayerTest addRef() {
    return (AssertDimensionsLayerTest) super.addRef();
  }

  public static class Basic extends AssertDimensionsLayerTest {

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
