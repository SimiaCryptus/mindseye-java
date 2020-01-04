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

public @com.simiacryptus.ref.lang.RefAware
class NthPowerActivationLayerTest {

  public static @com.simiacryptus.ref.lang.RefAware
  class InvPowerTest extends ActivationLayerTestBase {
    public InvPowerTest() {
      super(new NthPowerActivationLayer().setPower(-1));
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-4);
    }

    public static @SuppressWarnings("unused")
    InvPowerTest[] addRefs(InvPowerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(InvPowerTest::addRef)
          .toArray((x) -> new InvPowerTest[x]);
    }

    @Override
    public double random() {
      final double v = super.random();
      if (Math.abs(v) < 0.2)
        return random();
      return v;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    InvPowerTest addRef() {
      return (InvPowerTest) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class InvSqrtPowerTest extends ActivationLayerTestBase {
    public InvSqrtPowerTest() {
      super(new NthPowerActivationLayer().setPower(-0.5));
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-4);
    }

    public static @SuppressWarnings("unused")
    InvSqrtPowerTest[] addRefs(InvSqrtPowerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(InvSqrtPowerTest::addRef)
          .toArray((x) -> new InvSqrtPowerTest[x]);
    }

    @Override
    public double random() {
      final double v = super.random();
      if (Math.abs(v) < 0.2)
        return random();
      return v;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    InvSqrtPowerTest addRef() {
      return (InvSqrtPowerTest) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class NthPowerTest extends ActivationLayerTestBase {
    public NthPowerTest() {
      super(new NthPowerActivationLayer().setPower(Math.PI));
    }

    public static @SuppressWarnings("unused")
    NthPowerTest[] addRefs(NthPowerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(NthPowerTest::addRef)
          .toArray((x) -> new NthPowerTest[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    NthPowerTest addRef() {
      return (NthPowerTest) super.addRef();
    }

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

  public static @com.simiacryptus.ref.lang.RefAware
  class SquarePowerTest extends ActivationLayerTestBase {
    public SquarePowerTest() {
      super(new NthPowerActivationLayer().setPower(2));
    }

    public static @SuppressWarnings("unused")
    SquarePowerTest[] addRefs(SquarePowerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(SquarePowerTest::addRef)
          .toArray((x) -> new SquarePowerTest[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    SquarePowerTest addRef() {
      return (SquarePowerTest) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class ZeroPowerTest extends ActivationLayerTestBase {
    public ZeroPowerTest() {
      super(new NthPowerActivationLayer().setPower(0));
    }

    public static @SuppressWarnings("unused")
    ZeroPowerTest[] addRefs(ZeroPowerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(ZeroPowerTest::addRef)
          .toArray((x) -> new ZeroPowerTest[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    ZeroPowerTest addRef() {
      return (ZeroPowerTest) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new MeanSqLossLayer();
    }
  }

}
