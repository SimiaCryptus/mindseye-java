/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.notebook.NotebookOutput;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Activation key apply base.
 */
public abstract class ActivationLayerTestBase extends com.simiacryptus.mindseye.layers.ActivationLayerTestBase {
  /**
   * Instantiates a new Activation key apply base.
   *
   * @param layer the key
   */
  public ActivationLayerTestBase(Layer layer) {
    super(layer);
  }

  @Override
  protected Layer lossLayer() {
    return new EntropyLossLayer();
  }
}
