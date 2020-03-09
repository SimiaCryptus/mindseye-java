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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public final class NthPowerActivationLayer extends LayerBase {

  private double power = 1.0;

  public NthPowerActivationLayer() {
  }

  protected NthPowerActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    power = id.get("power").getAsDouble();
  }

  public NthPowerActivationLayer(double power) {
    this.power = power;
  }

  public double getPower() {
    return power;
  }

  public void setPower(double power) {
    this.power = power;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static NthPowerActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new NthPowerActivationLayer(json);
  }

  private static void nthPower(final double power, @Nonnull final Tensor input, final double[] inputData,
                               final double[] gradientData, final double[] outputData) {
    for (int i = 0; i < input.length(); i++) {
      final double x = inputData[i];
      final boolean isZero = Math.abs(x) < 1e-20;
      double d = isZero ? 0.0 : power * Math.pow(x, power - 1);
      double f = isZero ? 0.0 : Math.pow(x, power);
      if (!Double.isFinite(d)) {
        d = 0.0;
      }
      if (!Double.isFinite(f)) {
        f = 0.0;
      }
      gradientData[i] = d;
      outputData[i] = f;
    }
    input.freeRef();
  }

  private static void square(@Nonnull final Tensor input, final double[] inputData, final double[] gradientData,
                             final double[] outputData) {
    for (int i = 0; i < input.length(); i++) {
      final double x = inputData[i];
      gradientData[i] = 2 * x;
      outputData[i] = x * x;
    }
    input.freeRef();
  }

  private static void squareRoot(@Nonnull final Tensor input, final double[] inputData, final double[] gradientData,
                                 final double[] outputData) {
    for (int i = 0; i < input.length(); i++) {
      final double x = inputData[i];
      final boolean isZero = Math.abs(x) < 1e-20;
      final double power = 0.5;
      final double v = Math.pow(x, power);
      double d = isZero ? 0.0 : power / v;
      double f = isZero ? 0.0 : v;
      if (!Double.isFinite(d)) {
        d = 0.0;
      }
      if (!Double.isFinite(f)) {
        f = 0.0;
      }
      gradientData[i] = d;
      outputData[i] = f;
    }
    input.freeRef();
  }

  private static void unity(@Nonnull final Tensor input, final double[] inputData, final double[] gradientData,
                            final double[] outputData) {
    for (int i = 0; i < input.length(); i++) {
      gradientData[i] = 0;
      outputData[i] = 1;
    }
    input.freeRef();
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inData = in0.getData();
    final int itemCnt = inData.length();
    assert 0 < itemCnt;
    @Nonnull final Tensor inputGradientA[] = new Tensor[itemCnt];
    TensorArray data = fwd(inData, itemCnt, inputGradientA);
    boolean alive = 0.0 != power && in0.isAlive();
    Result.Accumulator accumulator = new Accumulator(inputGradientA, itemCnt, in0.getAccumulator(), in0.isAlive());
    in0.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("power", power);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  NthPowerActivationLayer addRef() {
    return (NthPowerActivationLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inData, int itemCnt, @RefIgnore Tensor[] inputGradientA) {
    return new TensorArray(RefIntStream.range(0, itemCnt).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable final Tensor input = inData.get(dataIndex);
          @Nonnull final Tensor output = new Tensor(inData.getDimensions());
          @Nonnull final Tensor gradient = new Tensor(input.length());
          @Nullable final double[] inputData = input.getData();
          @Nullable final double[] gradientData = gradient.getData();
          @Nullable final double[] outputData = output.getData();
          RefUtil.set(inputGradientA, dataIndex, gradient);
          if (power == 2) {
            NthPowerActivationLayer.square(input.addRef(), inputData, gradientData,
                outputData);
          } else if (power == 0.5) {
            NthPowerActivationLayer.squareRoot(input.addRef(), inputData, gradientData,
                outputData);
          } else if (power == 0.0) {
            NthPowerActivationLayer.unity(input.addRef(), inputData, gradientData,
                outputData);
          } else {
            NthPowerActivationLayer.nthPower(power, input.addRef(), inputData, gradientData,
                outputData);
          }
          input.freeRef();
          return output;
        }, inData)).toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final Tensor[] inputGradientA;
    private final int itemCnt;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(Tensor[] inputGradientA, int itemCnt, Result.Accumulator accumulator, boolean alive) {
      this.inputGradientA = inputGradientA;
      this.itemCnt = itemCnt;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive) {
        this.accumulator.accept(buffer, new TensorArray(RefIntStream.range(0, itemCnt).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(data.getDimensions());
              @Nullable final Tensor tensor = data.get(dataIndex);
              @Nullable
              double[] tensorData = tensor.getData();
              @Nullable final double[] gradientData = inputGradientA[dataIndex].getData();
              RefIntStream.range(0, passback.length()).forEach(RefUtil.wrapInterface(i -> {
                final double v = gradientData[i];
                if (Double.isFinite(v)) {
                  passback.set(i, tensorData[i] * v);
                }
              }, passback.addRef()));
              tensor.freeRef();
              return passback;
            }, data)).toArray(Tensor[]::new)));
      } else {
        data.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      RefUtil.freeRef(inputGradientA);
    }
  }
}
