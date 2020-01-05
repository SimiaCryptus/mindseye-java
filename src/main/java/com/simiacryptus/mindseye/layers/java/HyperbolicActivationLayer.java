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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class HyperbolicActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(HyperbolicActivationLayer.class);
  @Nullable
  private final Tensor weights;
  private int negativeMode = 1;

  public HyperbolicActivationLayer() {
    super();
    {
      Tensor temp_16_0001 = new Tensor(2);
      weights = temp_16_0001 == null ? null : temp_16_0001.addRef();
      if (null != temp_16_0001)
        temp_16_0001.freeRef();
    }
    RefUtil.freeRef(weights.set(0, 1.));
    RefUtil.freeRef(weights.set(1, 1.));
  }

  protected HyperbolicActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    {
      Tensor temp_16_0002 = Tensor.fromJson(json.get("weights"), resources);
      weights = temp_16_0002 == null ? null : temp_16_0002.addRef();
      if (null != temp_16_0002)
        temp_16_0002.freeRef();
    }
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
    RefUtil.freeRef(weights.set(0, 1 / scale));
    RefUtil.freeRef(weights.set(1, 1 / scale));
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static HyperbolicActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new HyperbolicActivationLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  HyperbolicActivationLayer[] addRefs(HyperbolicActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayer::addRef)
        .toArray((x) -> new HyperbolicActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  HyperbolicActivationLayer[][] addRefs(HyperbolicActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(HyperbolicActivationLayer::addRefs)
        .toArray((x) -> new HyperbolicActivationLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    final int itemCnt = indata.length();
    final HyperbolicActivationLayer hyperbolicActivationLayer = HyperbolicActivationLayer.this.addRef();
    try {
      try {
        try {
          return new Result(
              new TensorArray(RefIntStream.range(0, itemCnt).mapToObj(RefUtil.wrapInterface(
                  (IntFunction<? extends Tensor>) dataIndex -> {
                    @Nullable final Tensor input = indata.get(dataIndex);
                    Tensor temp_16_0005 = input.map(v -> {
                      final int sign = v < 0 ? negativeMode : 1;
                      final double a = Math.max(0, weights.get(v < 0 ? 1 : 0));
                      return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
                    });
                    if (null != input)
                      input.freeRef();
                    return temp_16_0005;
                  }, indata == null ? null : indata.addRef())).toArray(i -> new Tensor[i])),
              new Result.Accumulator() {
                {
                  Result.addRefs(inObj);
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (!HyperbolicActivationLayer.this.isFrozen()) {
                    RefIntStream.range(0, delta.length()).forEach(
                        RefUtil.wrapInterface(dataIndex -> {
                              @Nullable
                              Tensor deltaI = delta.get(dataIndex);
                              @Nullable
                              Tensor inputI = indata.get(dataIndex);
                              @Nullable final double[] deltaData = deltaI.getData();
                              if (null != deltaI)
                                deltaI.freeRef();
                              @Nullable final double[] inputData = inputI.getData();
                              if (null != inputI)
                                inputI.freeRef();
                              @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
                              for (int i = 0; i < deltaData.length; i++) {
                                final double d = deltaData[i];
                                final double x = inputData[i];
                                final int sign = x < 0 ? negativeMode : 1;
                                final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
                                weightDelta.add(x < 0 ? 1 : 0, -sign * d / (a * a * Math.sqrt(1 + Math.pow(a * x, 2))));
                              }
                              Delta<UUID> temp_16_0007 = buffer
                                  .get(hyperbolicActivationLayer.getId(), weights.getData());
                              RefUtil.freeRef(temp_16_0007.addInPlace(weightDelta.getData()));
                              if (null != temp_16_0007)
                                temp_16_0007.freeRef();
                              weightDelta.freeRef();
                            }, delta == null ? null : delta.addRef(),
                            hyperbolicActivationLayer == null ? null : hyperbolicActivationLayer.addRef(),
                            buffer == null ? null : buffer.addRef(), indata == null ? null : indata.addRef()));
                  }
                  if (inObj[0].isAlive()) {
                    @Nonnull
                    TensorArray tensorArray = new TensorArray(
                        RefIntStream.range(0, delta.length()).mapToObj(RefUtil.wrapInterface(
                            (IntFunction<? extends Tensor>) dataIndex -> {
                              @Nullable
                              Tensor inputTensor = indata.get(dataIndex);
                              Tensor deltaTensor = delta.get(dataIndex);
                              @Nullable final double[] deltaData = deltaTensor.getData();
                              if (null != deltaTensor)
                                deltaTensor.freeRef();
                              @Nonnull final int[] dims = indata.getDimensions();
                              @Nonnull final Tensor passback = new Tensor(dims);
                              for (int i = 0; i < passback.length(); i++) {
                                final double x = inputTensor.getData()[i];
                                final double d = deltaData[i];
                                final int sign = x < 0 ? negativeMode : 1;
                                final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
                                RefUtil
                                    .freeRef(passback.set(i, sign * d * a * x / Math.sqrt(1 + a * x * a * x)));
                              }
                              if (null != inputTensor)
                                inputTensor.freeRef();
                              return passback;
                            }, delta == null ? null : delta.addRef(), indata == null ? null : indata.addRef()))
                            .toArray(i -> new Tensor[i]));
                    inObj[0].accumulate(buffer == null ? null : buffer.addRef(),
                        tensorArray == null ? null : tensorArray);
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused")
                void _free() {
                  if (null != inObj)
                    ReferenceCounting.freeRefs(inObj);
                }
              }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return inObj[0].isAlive() || !isFrozen();
            }

            public void _free() {
              if (null != inObj)
                ReferenceCounting.freeRefs(inObj);
            }
          };
        } finally {
          if (null != inObj)
            ReferenceCounting.freeRefs(inObj);
        }
      } finally {
        if (null != hyperbolicActivationLayer)
          hyperbolicActivationLayer.freeRef();
      }
    } finally {
      if (null != indata)
        indata.freeRef();
    }

  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson(resources, dataSerializer));
    json.addProperty("negativeMode", negativeMode);
    return json;
  }

  @Nonnull
  public HyperbolicActivationLayer setModeAsymetric() {
    negativeMode = 0;
    return this.addRef();
  }

  @Nonnull
  public HyperbolicActivationLayer setModeEven() {
    negativeMode = 1;
    return this.addRef();
  }

  @Nonnull
  public HyperbolicActivationLayer setModeOdd() {
    negativeMode = -1;
    return this.addRef();
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
  HyperbolicActivationLayer addRef() {
    return (HyperbolicActivationLayer) super.addRef();
  }

}
