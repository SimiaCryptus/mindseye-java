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
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class ReshapeLayerTest extends LayerTestBase {

  private final int[] outputDims;
  private final int[] inputDims;

  protected ReshapeLayerTest(int[] inputDims, int[] outputDims) {
    this.inputDims = inputDims;
    this.outputDims = outputDims;
  }

  public static @SuppressWarnings("unused") ReshapeLayerTest[] addRefs(ReshapeLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReshapeLayerTest::addRef)
        .toArray((x) -> new ReshapeLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ReshapeLayerTest[][] addRefs(ReshapeLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReshapeLayerTest::addRefs)
        .toArray((x) -> new ReshapeLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { inputDims };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ReshapeLayer(outputDims);
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ReshapeLayerTest addRef() {
    return (ReshapeLayerTest) super.addRef();
  }

  public static class Basic extends ReshapeLayerTest {
    public Basic() {
      super(new int[] { 6, 6, 1 }, new int[] { 1, 1, 36 });
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

  public static class Basic1 extends ReshapeLayerTest {
    public Basic1() {
      super(new int[] { 1, 1, 32 }, new int[] { 1, 1, 32 });
    }

    public static @SuppressWarnings("unused") Basic1[] addRefs(Basic1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic1::addRef).toArray((x) -> new Basic1[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic1 addRef() {
      return (Basic1) super.addRef();
    }
  }

  public static class Big0 extends Big {
    public Big0() {
      super(256);
    }

    public static @SuppressWarnings("unused") Big0[] addRefs(Big0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big0::addRef).toArray((x) -> new Big0[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Big0 addRef() {
      return (Big0) super.addRef();
    }

  }

  public static class Big1 extends Big {
    public Big1() {
      super(new int[] { 4, 4, 256 }, new int[] { 1, 1, 2 * 2048 });
    }

    public static @SuppressWarnings("unused") Big1[] addRefs(Big1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big1::addRef).toArray((x) -> new Big1[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Big1 addRef() {
      return (Big1) super.addRef();
    }
  }

  public static class Big2 extends Big {
    public Big2() {
      super(new int[] { 1, 1, 2 * 2048 }, new int[] { 4, 4, 256 });
    }

    public static @SuppressWarnings("unused") Big2[] addRefs(Big2[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big2::addRef).toArray((x) -> new Big2[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Big2 addRef() {
      return (Big2) super.addRef();
    }
  }

  public abstract static class Big extends ReshapeLayerTest {

    public Big(int size) {
      this(new int[] { 1, 1, size }, new int[] { 1, 1, size });
    }

    public Big(int[] inputDims, int[] outputDims) {
      super(inputDims, outputDims);
      validateDifferentials = false;
    }

    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution)
        return null;
      return (new BatchingTester(1e-2, true) {
        @Override
        public double getRandom() {
          return random();
        }

        public @SuppressWarnings("unused") void _free() {
        }
      }).setBatchSize(5);
    }

    @Nullable
    @Override
    protected ComponentTest<ToleranceStatistics> getJsonTester() {
      logger.warn("Disabled Json Test");
      return null;
      //return super.getJsonTester();
    }

    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
      //return super.getPerformanceTester();
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    public static @SuppressWarnings("unused") Big[] addRefs(Big[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Big::addRef).toArray((x) -> new Big[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Big addRef() {
      return (Big) super.addRef();
    }
  }
}
