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
import com.simiacryptus.mindseye.layers.MetaLayerTestBase;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class NormalizationMetaLayerTest extends MetaLayerTestBase {
  @Override
  public ComponentTest<ToleranceStatistics> getDerivativeTester() {
    return null;
    //return new BatchDerivativeTester(1e-2, 1e-5, 10);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 3 } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new NormalizationMetaLayer();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { 10 } };
  }

  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  public static @com.simiacryptus.ref.lang.RefAware class Basic extends NormalizationMetaLayerTest {

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

  public @Override @SuppressWarnings("unused") NormalizationMetaLayerTest addRef() {
    return (NormalizationMetaLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") NormalizationMetaLayerTest[] addRefs(NormalizationMetaLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(NormalizationMetaLayerTest::addRef)
        .toArray((x) -> new NormalizationMetaLayerTest[x]);
  }

  public static @SuppressWarnings("unused") NormalizationMetaLayerTest[][] addRefs(
      NormalizationMetaLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(NormalizationMetaLayerTest::addRefs)
        .toArray((x) -> new NormalizationMetaLayerTest[x][]);
  }

}
