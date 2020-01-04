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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class HyperbolicActivationLayer extends LayerBase {

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

  protected HyperbolicActivationLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
    negativeMode = json.getAsJsonPrimitive("negativeMode").getAsInt();
  }

  public double getScaleL() {
    return 1 / weights.get(1);
  }

  public double getScaleR() {
    return 1 / weights.get(0);
  }

  @Nonnull
  public HyperbolicActivationLayer setScale(final double scale) {
    weights.set(0, 1 / scale);
    weights.set(1, 1 / scale);
    return this;
  }

  @SuppressWarnings("unused")
  public static HyperbolicActivationLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new HyperbolicActivationLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    final int itemCnt = indata.length();
    return new Result(
        new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToObj(dataIndex -> {
          @Nullable
          final Tensor input = indata.get(dataIndex);
          return input.map(v -> {
            final int sign = v < 0 ? negativeMode : 1;
            final double a = Math.max(0, weights.get(v < 0 ? 1 : 0));
            return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
          });
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (!isFrozen()) {
            com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).forEach(dataIndex -> {
              @Nullable
              Tensor deltaI = delta.get(dataIndex);
              @Nullable
              Tensor inputI = indata.get(dataIndex);
              @Nullable
              final double[] deltaData = deltaI.getData();
              @Nullable
              final double[] inputData = inputI.getData();
              @Nonnull
              final Tensor weightDelta = new Tensor(weights.getDimensions());
              for (int i = 0; i < deltaData.length; i++) {
                final double d = deltaData[i];
                final double x = inputData[i];
                final int sign = x < 0 ? negativeMode : 1;
                final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
                weightDelta.add(x < 0 ? 1 : 0, -sign * d / (a * a * Math.sqrt(1 + Math.pow(a * x, 2))));
              }
              buffer.get(HyperbolicActivationLayer.this.getId(), weights.getData()).addInPlace(weightDelta.getData());
            });
          }
          if (inObj[0].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
                  @Nullable
                  Tensor inputTensor = indata.get(dataIndex);
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nullable
                  final double[] deltaData = deltaTensor.getData();
                  @Nonnull
                  final int[] dims = indata.getDimensions();
                  @Nonnull
                  final Tensor passback = new Tensor(dims);
                  for (int i = 0; i < passback.length(); i++) {
                    final double x = inputTensor.getData()[i];
                    final double d = deltaData[i];
                    final int sign = x < 0 ? negativeMode : 1;
                    final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
                    passback.set(i, sign * d * a * x / Math.sqrt(1 + a * x * a * x));
                  }
                  return passback;
                }).toArray(i -> new Tensor[i]));
            inObj[0].accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };

  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson(resources, dataSerializer));
    json.addProperty("negativeMode", negativeMode);
    return json;
  }

  @Nonnull
  public HyperbolicActivationLayer setModeAsymetric() {
    negativeMode = 0;
    return this;
  }

  @Nonnull
  public HyperbolicActivationLayer setModeEven() {
    negativeMode = 1;
    return this;
  }

  @Nonnull
  public HyperbolicActivationLayer setModeOdd() {
    negativeMode = -1;
    return this;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(weights.getData());
  }

  public void _free() {
    super._free();
  }

  public @Override @SuppressWarnings("unused") HyperbolicActivationLayer addRef() {
    return (HyperbolicActivationLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") HyperbolicActivationLayer[] addRefs(HyperbolicActivationLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayer::addRef)
        .toArray((x) -> new HyperbolicActivationLayer[x]);
  }

  public static @SuppressWarnings("unused") HyperbolicActivationLayer[][] addRefs(HyperbolicActivationLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayer::addRefs)
        .toArray((x) -> new HyperbolicActivationLayer[x][]);
  }

}
