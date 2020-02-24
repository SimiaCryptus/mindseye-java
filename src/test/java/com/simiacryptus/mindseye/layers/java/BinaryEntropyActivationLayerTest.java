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

import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;

import javax.annotation.Nonnull;

public abstract class BinaryEntropyActivationLayerTest extends ActivationLayerTestBase {
  public BinaryEntropyActivationLayerTest() {
    super(new BinaryEntropyActivationLayer());
  }

  @Override
  public double random() {
    return 0.1 * Math.random() + 1.0;
  }

  @Nonnull
  @Override
  public RefDoubleStream scan() {
    return RefIntStream.range(50, 450).mapToDouble(x -> x * 1.0 / 500.0);
  }

  //  /**
  //   * Basic Test
  //   */
  //  public static class Basic extends BinaryEntropyActivationLayerTest {
  //  }

}
