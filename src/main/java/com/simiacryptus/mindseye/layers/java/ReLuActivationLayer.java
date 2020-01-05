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
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;

@SuppressWarnings("serial")
public @RefAware
class ReLuActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  @Nullable
  private final Tensor weights;

  public ReLuActivationLayer() {
    super();
    weights = new Tensor(1);
    weights.set(0, 1.);
    this.frozen = true;
  }

  protected ReLuActivationLayer(@Nonnull final JsonObject json,
                                Map<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  protected double getMobility() {
    return 1;
  }

  @Nonnull
  public ReLuActivationLayer setWeight(final double data) {
    weights.set(0, data);
    return this;
  }

  @Nonnull
  public ReLuActivationLayer setWeights(@Nonnull final DoubleSupplier f) {
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @SuppressWarnings("unused")
  public static ReLuActivationLayer fromJson(@Nonnull final JsonObject json,
                                             Map<CharSequence, byte[]> rs) {
    return new ReLuActivationLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ReLuActivationLayer[] addRefs(ReLuActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReLuActivationLayer::addRef)
        .toArray((x) -> new ReLuActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ReLuActivationLayer[][] addRefs(ReLuActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReLuActivationLayer::addRefs)
        .toArray((x) -> new ReLuActivationLayer[x][]);
  }

  @Nonnull
  public ReLuActivationLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, weights.getData());
    return this;
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0];
    final TensorList indata = input.getData();
    final int itemCnt = indata.length();
    return new Result(
        new TensorArray(RefIntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
          @Nullable
          Tensor tensorElement = indata.get(dataIndex);
          @Nonnull final Tensor tensor = tensorElement.multiply(weights.get(0));
          @Nullable final double[] outputData = tensor.getData();
          for (int i = 0; i < outputData.length; i++) {
            if (outputData[i] < 0) {
              outputData[i] = 0;
            }
          }
          return tensor;
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        RefIntStream.range(0, delta.length()).parallel().forEach(dataIndex -> {
          @Nullable
          Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = deltaTensor.getData();
          @Nullable
          Tensor inputTensor = indata.get(dataIndex);
          @Nullable final double[] inputData = inputTensor.getData();
          @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
          @Nullable final double[] weightDeltaData = weightDelta.getData();
          for (int i = 0; i < deltaData.length; i++) {
            weightDeltaData[0] += inputData[i] < 0 ? 0 : deltaData[i] * inputData[i];
          }
          buffer.get(ReLuActivationLayer.this.getId(), weights.getData()).addInPlace(weightDeltaData);
        });
      }
      if (input.isAlive()) {
        final double weight = weights.getData()[0];
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            RefIntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
              @Nullable
              Tensor deltaTensor = delta.get(dataIndex);
              @Nullable final double[] deltaData = deltaTensor.getData();
              @Nullable
              Tensor inTensor = indata.get(dataIndex);
              @Nullable final double[] inputData = inTensor.getData();
              @Nonnull final int[] dims = inTensor.getDimensions();
              @Nonnull final Tensor passback = new Tensor(dims);
              for (int i = 0; i < passback.length(); i++) {
                passback.set(i, inputData[i] < 0 ? 0 : deltaData[i] * weight);
              }
              return passback;
            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
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
  ReLuActivationLayer addRef() {
    return (ReLuActivationLayer) super.addRef();
  }

}
