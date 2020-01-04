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

import java.util.HashMap;
import com.simiacryptus.ref.wrappers.RefHashMap;

public abstract @com.simiacryptus.ref.lang.RefAware class HyperbolicActivationLayerTest
    extends ActivationLayerTestBase {
  public HyperbolicActivationLayerTest() {
    super(new HyperbolicActivationLayer());
  }

  @Override
  protected com.simiacryptus.ref.wrappers.RefHashMap<Tensor[], Tensor> getReferenceIO() {
    final com.simiacryptus.ref.wrappers.RefHashMap<Tensor[], Tensor> map = super.getReferenceIO();
    map.put(new Tensor[] { new Tensor(0.0) }, new Tensor(0.0));
    return map;
  }

  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return new TrainingTester() {

      @Override
      protected Layer lossLayer() {
        return HyperbolicActivationLayerTest.this.lossLayer();
      }

      public @SuppressWarnings("unused") void _free() {
      }
    }.setRandomizationMode(TrainingTester.RandomizationMode.Random);
  }

  public static @com.simiacryptus.ref.lang.RefAware class Basic extends HyperbolicActivationLayerTest {
    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }

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

  public @Override @SuppressWarnings("unused") HyperbolicActivationLayerTest addRef() {
    return (HyperbolicActivationLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") HyperbolicActivationLayerTest[] addRefs(
      HyperbolicActivationLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayerTest::addRef)
        .toArray((x) -> new HyperbolicActivationLayerTest[x]);
  }

  public static @SuppressWarnings("unused") HyperbolicActivationLayerTest[][] addRefs(
      HyperbolicActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayerTest::addRefs)
        .toArray((x) -> new HyperbolicActivationLayerTest[x][]);
  }

}
