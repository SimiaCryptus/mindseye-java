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
class ImgTileAssemblyLayerTest extends LayerTestBase {

  public ImgTileAssemblyLayerTest() {
    validateBatchExecution = false;
  }

  public static @SuppressWarnings("unused")
  ImgTileAssemblyLayerTest[] addRefs(ImgTileAssemblyLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileAssemblyLayerTest::addRef)
        .toArray((x) -> new ImgTileAssemblyLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileAssemblyLayerTest[][] addRefs(ImgTileAssemblyLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileAssemblyLayerTest::addRefs)
        .toArray((x) -> new ImgTileAssemblyLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 1}, {1, 2, 1}, {2, 2, 1}, {1, 2, 1}, {2, 1, 1}, {1, 1, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgTileAssemblyLayer(2, 3);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgTileAssemblyLayerTest addRef() {
    return (ImgTileAssemblyLayerTest) super.addRef();
  }

  public static @RefAware
  class Basic extends ImgTileAssemblyLayerTest {

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
