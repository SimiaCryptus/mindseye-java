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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class LinearActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);
  @Nullable
  private final Tensor weights;

  public LinearActivationLayer() {
    super();
    weights = new Tensor(2);
    weights.set(0, 1.);
    weights.set(1, 0.);
  }

  protected LinearActivationLayer(@Nonnull final JsonObject json,
                                  Map<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  public double getBias() {
    return weights.get(1);
  }

  @Nonnull
  public LinearActivationLayer setBias(final double bias) {
    if (!Double.isFinite(bias))
      throw new IllegalArgumentException();
    weights.set(1, bias);
    return this;
  }

  @Nullable
  @Override
  public String getName() {
    String eqStr = isFrozen() ? "== " : "=> ";
    if (weights.get(0) == 1.0) {
      return eqStr + String.format("x + %.1e", weights.get(1)) + (isFrozen() ? "" : "!");
    } else if (weights.get(1) == 0.0) {
      return eqStr + String.format("%.1e x", weights.get(0)) + (isFrozen() ? "" : "!");
    } else {
      return eqStr + String.format("%.1e x + %.1e", weights.get(0), weights.get(1));
    }
  }

  public double getScale() {
    return weights.get(0);
  }

  @Nonnull
  public LinearActivationLayer setScale(final double scale) {
    if (!Double.isFinite(scale))
      throw new IllegalArgumentException();
    weights.set(0, scale);
    return this;
  }

  @SuppressWarnings("unused")
  public static LinearActivationLayer fromJson(@Nonnull final JsonObject json,
                                               Map<CharSequence, byte[]> rs) {
    return new LinearActivationLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  LinearActivationLayer[] addRefs(LinearActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LinearActivationLayer::addRef)
        .toArray((x) -> new LinearActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  LinearActivationLayer[][] addRefs(LinearActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LinearActivationLayer::addRefs)
        .toArray((x) -> new LinearActivationLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result in0 = inObj[0];
    final TensorList inData = in0.getData();
    final int itemCnt = inData.length();
    final double scale = weights.get(0);
    final double bias = weights.get(1);
    final LinearActivationLayer linearActivationLayer = LinearActivationLayer.this;
    return new Result(new TensorArray(RefIntStream.range(0, itemCnt)
        .mapToObj(dataIndex -> inData.get(dataIndex).map(v -> {
          final double r = scale * v + bias;
          return Double.isFinite(r) ? r : 0;
        })).toArray(i -> new Tensor[i])), new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList delta) {
        if (!LinearActivationLayer.this.isFrozen()) {
          RefIntStream.range(0, delta.length()).forEach(dataIndex -> {
            @Nullable
            Tensor deltaT = delta.get(dataIndex);
            @Nullable
            Tensor inputT = inData.get(dataIndex);
            @Nullable final double[] deltaData = deltaT.getData();
            @Nullable final double[] inputData = inputT.getData();
            @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
            for (int i = 0; i < deltaData.length; i++) {
              weightDelta.add(0, deltaData[i] * inputData[inputData.length == 1 ? 0 : i]);
              weightDelta.add(1, deltaData[i]);
            }
            buffer.get(linearActivationLayer.getId(), weights.getData()).addInPlace(weightDelta.getData());
          });
        }
        if (in0.isAlive()) {
          @Nonnull final TensorList tensorList = new TensorArray(
              RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
                @Nullable
                Tensor tensor = delta.get(dataIndex);
                @Nullable final double[] deltaData = tensor.getData();
                @Nonnull final Tensor passback = new Tensor(inData.getDimensions());
                for (int i = 0; i < passback.length(); i++) {
                  passback.set(i, deltaData[i] * weights.getData()[0]);
                }
                return passback;
              }).toArray(i -> new Tensor[i]));
          in0.accumulate(buffer, tensorList);
        }
      }
    }) {

      @Override
      public boolean isAlive() {
        return in0.isAlive() || !isFrozen();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(weights.getData());
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  LinearActivationLayer addRef() {
    return (LinearActivationLayer) super.addRef();
  }

}
