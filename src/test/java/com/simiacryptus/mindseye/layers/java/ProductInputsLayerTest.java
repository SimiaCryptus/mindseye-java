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
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.mindseye.test.unit.TrainingTester;

import javax.annotation.Nonnull;

public abstract class ProductInputsLayerTest extends LayerTestBase {
  @Nonnull
  @Override
  public Layer getLayer() {
    return new ProductInputsLayer();
  }

  @Override
  public TrainingTester getTrainingTester() {
    TrainingTester trainingTester = super.getTrainingTester();
    trainingTester.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    return trainingTester;
  }

  public static class N1Test extends ProductInputsLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3}, {1}};
    }

  }

  public static class NNNTest extends ProductInputsLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3}, {3}, {3}};
    }

  }

  public static class NNTest extends ProductInputsLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3}, {3}};
    }

  }
}
