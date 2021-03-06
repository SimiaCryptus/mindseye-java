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
import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import javax.annotation.Nonnull;

/**
 * The type Log activation layer test.
 */
public abstract class LogActivationLayerTest extends ActivationLayerTestBase {
  /**
   * Instantiates a new Log activation layer test.
   */
  public LogActivationLayerTest() {
    super(new LogActivationLayer());
  }

  @Override
  public SingleDerivativeTester getDerivativeTester() {
    return new SingleDerivativeTester(1e-2, 1e-8);
  }

  /**
   * The type Basic.
   */
  public static class Basic extends LogActivationLayerTest {

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

}
