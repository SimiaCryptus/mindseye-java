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
 * The type Nth power activation layer test.
 */
public class NthPowerActivationLayerTest {

  /**
   * The type Inv power test.
   */
  public static class InvPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Inv power test.
     */
    public InvPowerTest() {
      super(new NthPowerActivationLayer(-1));
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-4);
    }

    @Override
    public double random() {
      final double v = super.random();
      if (Math.abs(v) < 0.2)
        return random();
      return v;
    }

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  /**
   * The type Inv sqrt power test.
   */
  public static class InvSqrtPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Inv sqrt power test.
     */
    public InvSqrtPowerTest() {
      super(new NthPowerActivationLayer(-0.5));
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-4);
    }

    @Override
    public double random() {
      final double v = super.random();
      if (Math.abs(v) < 0.2)
        return random();
      return v;
    }

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  /**
   * The type Nth power test.
   */
  public static class NthPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Nth power test.
     */
    public NthPowerTest() {
      super(new NthPowerActivationLayer(Math.PI));
    }

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  //  /**
  //   * Tests x^1/2 aka sqrt(x)
  //   */
  //  public static class SqrtPowerTest extends ActivationLayerTestBase {
  //    /**
  //     * Instantiates a new Sqrt power apply.
  //     */
  //    public SqrtPowerTest() {
  //      super(new NthPowerActivationLayer().setPower(0.5));
  //    }
  //  }

  /**
   * The type Square power test.
   */
  public static class SquarePowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Square power test.
     */
    public SquarePowerTest() {
      super(new NthPowerActivationLayer(2));
    }

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  /**
   * The type Zero power test.
   */
  public static class ZeroPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Zero power test.
     */
    public ZeroPowerTest() {
      super(new NthPowerActivationLayer(0));
    }

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

}
