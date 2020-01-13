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

public abstract class FullyConnectedReferenceLayerTest extends LayerTestBase {
  @Nonnull
  private final int[] outputDims;
  private final int[] inputDims;
  @Nonnull
  private final FullyConnectedReferenceLayer layer;

  public FullyConnectedReferenceLayerTest(int[] inputDims, @Nonnull int[] outputDims) {
    this.outputDims = outputDims;
    this.inputDims = inputDims;
    FullyConnectedReferenceLayer temp_00_0002 = new FullyConnectedReferenceLayer(getSmallDims(new Random())[0],
        outputDims);
    FullyConnectedReferenceLayer temp_00_0001 = temp_00_0002.set(i -> random());
    if (null != temp_00_0002)
      temp_00_0002.freeRef();
    this.layer = temp_00_0001 == null ? null : temp_00_0001.addRef();
    if (null != temp_00_0001)
      temp_00_0001.freeRef();
  }

  public static @SuppressWarnings("unused") FullyConnectedReferenceLayerTest[] addRefs(
      FullyConnectedReferenceLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedReferenceLayerTest::addRef)
        .toArray((x) -> new FullyConnectedReferenceLayerTest[x]);
  }

  public static @SuppressWarnings("unused") FullyConnectedReferenceLayerTest[][] addRefs(
      FullyConnectedReferenceLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedReferenceLayerTest::addRefs)
        .toArray((x) -> new FullyConnectedReferenceLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { inputDims };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer == null ? null : layer.addRef();
  }

  public @SuppressWarnings("unused") void _free() {
    layer.freeRef();
  }

  public @Override @SuppressWarnings("unused") FullyConnectedReferenceLayerTest addRef() {
    return (FullyConnectedReferenceLayerTest) super.addRef();
  }

  public static class Basic extends FullyConnectedReferenceLayerTest {
    public Basic() {
      super(new int[] { 2 }, new int[] { 2 });
    }

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }
  }

  public static class Image extends FullyConnectedReferenceLayerTest {
    public Image() {
      super(new int[] { 3, 3, 3 }, new int[] { 2, 2, 4 });
    }

    public static @SuppressWarnings("unused") Image[] addRefs(Image[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Image::addRef).toArray((x) -> new Image[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Image addRef() {
      return (Image) super.addRef();
    }
  }

}
