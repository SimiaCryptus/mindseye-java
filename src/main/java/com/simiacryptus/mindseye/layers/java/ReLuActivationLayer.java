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
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class ReLuActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  @Nullable
  private final Tensor weights;

  public ReLuActivationLayer() {
    super();
    Tensor temp_23_0001 = new Tensor(1);
    weights = temp_23_0001.addRef();
    temp_23_0001.freeRef();
    weights.set(0, 1.);

    this.frozen = true;
  }

  protected ReLuActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor temp_23_0002 = Tensor.fromJson(json.get("weights"), resources);
    weights = temp_23_0002 == null ? null : temp_23_0002.addRef();
    if (null != temp_23_0002)
      temp_23_0002.freeRef();
  }

  protected double getMobility() {
    return 1;
  }

  public void setWeight(double data) {
    assert weights != null;
    weights.set(0, data);
  }

  public void setWeights(@Nonnull DoubleSupplier f) {
    assert weights != null;
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ReLuActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ReLuActivationLayer(json, rs);
  }

  public void addWeights(@Nonnull DoubleSupplier f) {
    assert weights != null;
    Util.add(f, weights.getData());
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result input = inObj[0].addRef();
    RefUtil.freeRefs(inObj);
    final TensorList indata = input.getData();
    final int itemCnt = indata.length();
    final ReLuActivationLayer reLuActivationLayer = ReLuActivationLayer.this.addRef();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          reLuActivationLayer.addRef();
          indata.addRef();
          weights.addRef();
          input.addRef();
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (!ReLuActivationLayer.this.isFrozen()) {
            RefIntStream.range(0, delta.length()).parallel().forEach(RefUtil.wrapInterface(dataIndex -> {
                  @Nullable
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nullable final double[] deltaData = deltaTensor.getData();
                  deltaTensor.freeRef();
                  @Nullable
                  Tensor inputTensor = indata.get(dataIndex);
                  @Nullable final double[] inputData = inputTensor.getData();
                  inputTensor.freeRef();
                  assert weights != null;
                  @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
                  @Nullable final double[] weightDeltaData = weightDelta.getData();
                  weightDelta.freeRef();
                  for (int i = 0; i < deltaData.length; i++) {
                    weightDeltaData[0] += inputData[i] < 0 ? 0 : deltaData[i] * inputData[i];
                  }
                  Delta<UUID> temp_23_0006 = buffer.get(reLuActivationLayer.getId(), weights.getData());
                  assert temp_23_0006 != null;
                  temp_23_0006.addInPlace(weightDeltaData);
                  temp_23_0006.freeRef();
                }, buffer.addRef(), delta.addRef(),
                indata.addRef(),
                reLuActivationLayer.addRef()));
          }
          if (input.isAlive()) {
            assert weights != null;
            final double weight = weights.getData()[0];
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nullable
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nullable final double[] deltaData = deltaTensor.getData();
                  deltaTensor.freeRef();
                  @Nullable
                  Tensor inTensor = indata.get(dataIndex);
                  @Nullable final double[] inputData = inTensor.getData();
                  @Nonnull final int[] dims = inTensor.getDimensions();
                  inTensor.freeRef();
                  @Nonnull final Tensor passback = new Tensor(dims);
                  for (int i = 0; i < passback.length(); i++) {
                    final double value = inputData[i] < 0 ? 0 : deltaData[i] * weight;
                    passback.set(i, value);
                  }
                  return passback;
                }, delta.addRef(), indata.addRef()))
                .toArray(i -> new Tensor[i]));
            input.accumulate(buffer.addRef(), tensorArray);
          }
          delta.freeRef();
          buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          reLuActivationLayer.freeRef();
          indata.freeRef();
          weights.freeRef();
          input.freeRef();
        }
      };
      TensorArray data = new TensorArray(RefIntStream.range(0, itemCnt).parallel()
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            @Nullable
            Tensor tensorElement = indata.get(dataIndex);
            assert weights != null;
            @Nonnull final Tensor tensor = tensorElement.multiply(weights.get(0));
            tensorElement.freeRef();
            @Nullable final double[] outputData = tensor.getData();
            for (int i = 0; i < outputData.length; i++) {
              if (outputData[i] < 0) {
                outputData[i] = 0;
              }
            }
            return tensor;
          }, indata.addRef())).toArray(i -> new Tensor[i]));
      return new Result(data, accumulator) {
        {
          input.freeRef();
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }

        @Override
        public void _free() {
          input.freeRef();
          super._free();
        }
      };
    } finally {
      reLuActivationLayer.freeRef();
      indata.freeRef();
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert weights != null;
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
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
  ReLuActivationLayer addRef() {
    return (ReLuActivationLayer) super.addRef();
  }

}
