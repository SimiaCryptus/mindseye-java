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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefHashMap;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public abstract class HyperbolicActivationLayerTest extends ActivationLayerTestBase {
  public HyperbolicActivationLayerTest() {
    super(new HyperbolicActivationLayer());
  }

  @Nullable
  @Override
  protected RefHashMap<Tensor[], Tensor> getReferenceIO() {
    final RefHashMap<Tensor[], Tensor> map = RefUtil.addRef(super.getReferenceIO());
    assert map != null;
    RefUtil.freeRef(map.put(new Tensor[]{new Tensor(0.0)}, new Tensor(0.0)));
    return map;
  }

  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    TrainingTester temp_69_0002 = new TrainingTester() {

      public @SuppressWarnings("unused")
      void _free() {
      }

      @Override
      protected Layer lossLayer() {
        return HyperbolicActivationLayerTest.this.lossLayer();
      }
    };
    TrainingTester temp_69_0001 = temp_69_0002.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    temp_69_0002.freeRef();
    return temp_69_0001;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  HyperbolicActivationLayerTest[] addRefs(
      @Nullable HyperbolicActivationLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayerTest::addRef)
        .toArray((x) -> new HyperbolicActivationLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  HyperbolicActivationLayerTest[][] addRefs(
      @Nullable HyperbolicActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayerTest::addRefs)
        .toArray((x) -> new HyperbolicActivationLayerTest[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  HyperbolicActivationLayerTest addRef() {
    return (HyperbolicActivationLayerTest) super.addRef();
  }

  public static class Basic extends HyperbolicActivationLayerTest {
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

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
