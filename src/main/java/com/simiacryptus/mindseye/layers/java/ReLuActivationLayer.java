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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
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
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class ReLuActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  @Nullable
  private final Tensor weights;

  public ReLuActivationLayer() {
    super();
    {
      Tensor temp_23_0001 = new Tensor(1);
      weights = temp_23_0001 == null ? null : temp_23_0001.addRef();
      if (null != temp_23_0001)
        temp_23_0001.freeRef();
    }
    RefUtil.freeRef(weights.set(0, 1.));
    this.frozen = true;
  }

  protected ReLuActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    {
      Tensor temp_23_0002 = Tensor.fromJson(json.get("weights"), resources);
      weights = temp_23_0002 == null ? null : temp_23_0002.addRef();
      if (null != temp_23_0002)
        temp_23_0002.freeRef();
    }
  }

  protected double getMobility() {
    return 1;
  }

  @Nonnull
  public ReLuActivationLayer setWeight(final double data) {
    RefUtil.freeRef(weights.set(0, data));
    return this.addRef();
  }

  @Nonnull
  public ReLuActivationLayer setWeights(@Nonnull final DoubleSupplier f) {
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static ReLuActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
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
    return this.addRef();
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0].addRef();
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    final TensorList indata = input.getData();
    final int itemCnt = indata.length();
    final ReLuActivationLayer reLuActivationLayer = ReLuActivationLayer.this.addRef();
    try {
      try {
        try {
          return new Result(new TensorArray(
              RefIntStream.range(0, itemCnt).parallel().mapToObj(RefUtil.wrapInterface(
                  (IntFunction<? extends Tensor>) dataIndex -> {
                    @Nullable
                    Tensor tensorElement = indata.get(dataIndex);
                    @Nonnull final Tensor tensor = tensorElement.multiply(weights.get(0));
                    if (null != tensorElement)
                      tensorElement.freeRef();
                    @Nullable final double[] outputData = tensor.getData();
                    for (int i = 0; i < outputData.length; i++) {
                      if (outputData[i] < 0) {
                        outputData[i] = 0;
                      }
                    }
                    return tensor;
                  }, indata == null ? null : indata.addRef())).toArray(i -> new Tensor[i])),
              new Result.Accumulator() {
                {
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (!ReLuActivationLayer.this.isFrozen()) {
                    RefIntStream.range(0, delta.length()).parallel().forEach(
                        RefUtil.wrapInterface(dataIndex -> {
                              @Nullable
                              Tensor deltaTensor = delta.get(dataIndex);
                              @Nullable final double[] deltaData = deltaTensor.getData();
                              if (null != deltaTensor)
                                deltaTensor.freeRef();
                              @Nullable
                              Tensor inputTensor = indata.get(dataIndex);
                              @Nullable final double[] inputData = inputTensor.getData();
                              if (null != inputTensor)
                                inputTensor.freeRef();
                              @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
                              @Nullable final double[] weightDeltaData = weightDelta.getData();
                              weightDelta.freeRef();
                              for (int i = 0; i < deltaData.length; i++) {
                                weightDeltaData[0] += inputData[i] < 0 ? 0 : deltaData[i] * inputData[i];
                              }
                              Delta<UUID> temp_23_0006 = buffer
                                  .get(reLuActivationLayer.getId(), weights.getData());
                              RefUtil.freeRef(temp_23_0006.addInPlace(weightDeltaData));
                              if (null != temp_23_0006)
                                temp_23_0006.freeRef();
                            }, buffer == null ? null : buffer.addRef(), delta == null ? null : delta.addRef(),
                            indata == null ? null : indata.addRef(),
                            reLuActivationLayer == null ? null : reLuActivationLayer.addRef()));
                  }
                  if (input.isAlive()) {
                    final double weight = weights.getData()[0];
                    @Nonnull
                    TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length()).parallel()
                        .mapToObj(RefUtil.wrapInterface(
                            (IntFunction<? extends Tensor>) dataIndex -> {
                              @Nullable
                              Tensor deltaTensor = delta.get(dataIndex);
                              @Nullable final double[] deltaData = deltaTensor.getData();
                              if (null != deltaTensor)
                                deltaTensor.freeRef();
                              @Nullable
                              Tensor inTensor = indata.get(dataIndex);
                              @Nullable final double[] inputData = inTensor.getData();
                              @Nonnull final int[] dims = inTensor.getDimensions();
                              if (null != inTensor)
                                inTensor.freeRef();
                              @Nonnull final Tensor passback = new Tensor(dims);
                              for (int i = 0; i < passback.length(); i++) {
                                RefUtil
                                    .freeRef(passback.set(i, inputData[i] < 0 ? 0 : deltaData[i] * weight));
                              }
                              return passback;
                            }, delta == null ? null : delta.addRef(), indata == null ? null : indata.addRef()))
                        .toArray(i -> new Tensor[i]));
                    input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused")
                void _free() {
                }
              }) {

            {
            }

            @Override
            public boolean isAlive() {
              return input.isAlive() || !isFrozen();
            }

            public void _free() {
            }

          };
        } finally {
          if (null != reLuActivationLayer)
            reLuActivationLayer.freeRef();
        }
      } finally {
        if (null != indata)
          indata.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
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
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  ReLuActivationLayer addRef() {
    return (ReLuActivationLayer) super.addRef();
  }

}
