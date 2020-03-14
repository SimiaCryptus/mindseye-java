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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class HyperbolicActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(HyperbolicActivationLayer.class);
  @Nullable
  private final Tensor weights;
  private int negativeMode = 1;

  public HyperbolicActivationLayer() {
    super();
    weights = new Tensor(2);
    weights.set(0, 1.);
    weights.set(1, 1.);
  }

  protected HyperbolicActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
    negativeMode = json.getAsJsonPrimitive("negativeMode").getAsInt();
  }

  public double getScaleL() {
    assert weights != null;
    return 1 / weights.get(1);
  }

  public double getScaleR() {
    assert weights != null;
    return 1 / weights.get(0);
  }

  public void setScale(double scale) {
    assert weights != null;
    weights.set(0, 1 / scale);
    weights.set(1, 1 / scale);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static HyperbolicActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new HyperbolicActivationLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final TensorList indata = inObj[0].getData();
    TensorArray data = fwd(indata);
    boolean alive = inObj[0].isAlive();
    final Result.Accumulator accumulator1 = inObj[0].getAccumulator();
    final boolean alive1 = inObj[0].isAlive();
    final @NotNull TensorList data1 = inObj[0].getData();
    Result.Accumulator accumulator = new Accumulator(negativeMode, weights.addRef(), getId(), isFrozen(), accumulator1, alive1, data1);
    RefUtil.freeRef(inObj);
    return new Result(data, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert weights != null;
    json.add("weights", weights.getJson(resources, dataSerializer));
    json.addProperty("negativeMode", negativeMode);
    return json;
  }

  public void setModeAsymetric() {
    negativeMode = 0;
  }

  public void setModeEven() {
    negativeMode = 1;
  }

  public void setModeOdd() {
    negativeMode = -1;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    assert weights != null;
    return RefArrays.asList(weights.getData());
  }

  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  HyperbolicActivationLayer addRef() {
    return (HyperbolicActivationLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList indata) {
    final int itemCnt = indata.length();
    return new TensorArray(RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable final Tensor input = indata.get(dataIndex);
          Tensor temp_16_0005 = input.map(v -> {
            final int sign = v < 0 ? negativeMode : 1;
            assert weights != null;
            final double a = Math.max(0, weights.get(v < 0 ? 1 : 0));
            return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
          });
          input.freeRef();
          return temp_16_0005;
        }, indata)).toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList indata;
    private boolean frozen;
    private UUID id;
    private int negativeMode;
    private Tensor weights;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(int negativeMode, Tensor weights, UUID id, boolean frozen, Result.Accumulator accumulator, boolean alive, @NotNull TensorList data) {
      this.indata = data;
      assert weights != null;
      this.frozen = frozen;
      this.id = id;
      this.negativeMode = negativeMode;
      this.weights = weights;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (!frozen) {
        RefIntStream.range(0, delta.length()).forEach(RefUtil.wrapInterface(dataIndex -> {
              @Nullable
              Tensor deltaI = delta.get(dataIndex);
              @Nullable
              Tensor inputI = indata.get(dataIndex);
              @Nullable final double[] deltaData = deltaI.getData();
              @Nullable final double[] inputData = inputI.getData();
              assert weights != null;
              @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
              for (int i = 0; i < deltaData.length; i++) {
                final double d = deltaData[i];
                final double x = inputData[i];
                final int sign = x < 0 ? negativeMode : 1;
                final double a = Math.max(0, weights.get(x < 0 ? 1 : 0));
                weightDelta.add(x < 0 ? 1 : 0, -sign * d / (a * a * Math.sqrt(1 + Math.pow(a * x, 2))));
              }
              deltaI.freeRef();
              inputI.freeRef();
              Delta<UUID> d = buffer.get(id, weights.getData());
              assert d != null;
              d.addInPlace(weightDelta.getData());
              d.freeRef();
              weightDelta.freeRef();
            }, delta.addRef(),
            weights.addRef(),
            buffer.addRef(), indata.addRef()));
      }
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable
              Tensor inputTensor = indata.get(dataIndex);
              Tensor deltaTensor = delta.get(dataIndex);
              @Nonnull final int[] dims = indata.getDimensions();
              @Nonnull final Tensor passback = new Tensor(dims);
              @Nullable final double[] deltaData = deltaTensor.getData();
              double[] tensorData = inputTensor.getData();
              for (int i = 0; i < passback.length(); i++) {
                final double x = tensorData[i];
                final double d = deltaData[i];
                final int sign = x < 0 ? negativeMode : 1;
                assert weights != null;
                final double a = Math.max(0, weights.get(x < 0 ? 1 : 0));
                passback.set(i, sign * d * a * x / Math.sqrt(1 + a * x * a * x));
              }
              deltaTensor.freeRef();
              inputTensor.freeRef();
              return passback;
            }, delta, indata.addRef()))
            .toArray(Tensor[]::new));
        try {
          this.accumulator.accept(buffer, tensorArray);
        } finally {
          this.accumulator.freeRef();
        }
      } else {
        buffer.freeRef();
        delta.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(accumulator);
      indata.freeRef();
      assert weights != null;
      weights.freeRef();
    }
  }
}
