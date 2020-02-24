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

public abstract class HyperbolicActivationLayerTest extends ActivationLayerTestBase {
  public HyperbolicActivationLayerTest() {
    super(new HyperbolicActivationLayer());
  }

  @Nullable
  @Override
  protected RefHashMap<Tensor[], Tensor> getReferenceIO() {
    final RefHashMap<Tensor[], Tensor> map = super.getReferenceIO();
    assert map != null;
    RefUtil.freeRef(map.put(new Tensor[]{new Tensor(0.0)}, new Tensor(0.0)));
    return map;
  }

  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    TrainingTester trainingTester = new TrainingTester() {

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
      }

      @Override
      protected Layer lossLayer() {
        return HyperbolicActivationLayerTest.this.lossLayer();
      }
    };
    trainingTester.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    return trainingTester;
  }

  public static class Basic extends HyperbolicActivationLayerTest {

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
