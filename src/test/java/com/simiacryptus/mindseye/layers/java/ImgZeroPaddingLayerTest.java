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

public abstract class ImgZeroPaddingLayerTest extends LayerTestBase {

  public ImgZeroPaddingLayerTest() {
    validateBatchExecution = false;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgZeroPaddingLayerTest[] addRefs(@Nullable ImgZeroPaddingLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgZeroPaddingLayerTest::addRef)
        .toArray((x) -> new ImgZeroPaddingLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgZeroPaddingLayerTest[][] addRefs(@Nullable ImgZeroPaddingLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgZeroPaddingLayerTest::addRefs)
        .toArray((x) -> new ImgZeroPaddingLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgZeroPaddingLayer(1, 1);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgZeroPaddingLayerTest addRef() {
    return (ImgZeroPaddingLayerTest) super.addRef();
  }

  public static class Basic extends ImgZeroPaddingLayerTest {

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
