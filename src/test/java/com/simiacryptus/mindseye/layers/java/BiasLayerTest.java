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

public abstract class BiasLayerTest extends LayerTestBase {

  private final int dimension;

  public BiasLayerTest(int dimension) {
    this.dimension = dimension;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BiasLayerTest[] addRefs(@Nullable BiasLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasLayerTest::addRef)
        .toArray((x) -> new BiasLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BiasLayerTest[][] addRefs(@Nullable BiasLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasLayerTest::addRefs)
        .toArray((x) -> new BiasLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{dimension}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    BiasLayer temp_75_0002 = new BiasLayer(dimension);
    BiasLayer temp_75_0001 = temp_75_0002.addWeights(this::random);
    temp_75_0002.freeRef();
    return temp_75_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BiasLayerTest addRef() {
    return (BiasLayerTest) super.addRef();
  }

  public static class Basic extends BiasLayerTest {
    public Basic() {
      super(5);
    }

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

  public static class Reducing extends BiasLayerTest {

    public Reducing() {
      super(5);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Reducing[] addRefs(@Nullable Reducing[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Reducing::addRef).toArray((x) -> new Reducing[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      BiasLayer temp_75_0004 = new BiasLayer(1);
      BiasLayer temp_75_0003 = temp_75_0004.addWeights(this::random);
      temp_75_0004.freeRef();
      return temp_75_0003;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Reducing addRef() {
      return (Reducing) super.addRef();
    }

  }

}
