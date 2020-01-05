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
class EntropyLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  public EntropyLossLayer() {
  }

  protected EntropyLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static EntropyLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new EntropyLossLayer(json);
  }

  public static @SuppressWarnings("unused")
  EntropyLossLayer[] addRefs(EntropyLossLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EntropyLossLayer::addRef)
        .toArray((x) -> new EntropyLossLayer[x]);
  }

  public static @SuppressWarnings("unused")
  EntropyLossLayer[][] addRefs(EntropyLossLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EntropyLossLayer::addRefs)
        .toArray((x) -> new EntropyLossLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final double zero_tol = 1e-12;
    TensorList indata = inObj[0].getData();
    @Nonnull final Tensor gradient[] = new Tensor[indata.length()];
    final double max_prob = 1.;
    try {
      try {
        try {
          return new Result(new TensorArray(
              RefIntStream.range(0, indata.length()).mapToObj(RefUtil.wrapInterface(
                  (IntFunction<? extends Tensor>) dataIndex -> {
                    @Nullable final Tensor l = indata.get(dataIndex);
                    TensorList temp_03_0006 = inObj[1].getData();
                    @Nullable final Tensor r = temp_03_0006.get(dataIndex);
                    if (null != temp_03_0006)
                      temp_03_0006.freeRef();
                    if (l.length() != r.length()) {
                      IllegalArgumentException temp_03_0004 = new IllegalArgumentException(
                          l.length() + " != " + r.length());
                      if (null != l)
                        l.freeRef();
                      if (null != r)
                        r.freeRef();
                      throw temp_03_0004;
                    }
                    @Nonnull final Tensor gradientTensor = new Tensor(l.getDimensions());
                    @Nullable final double[] gradientData = gradientTensor.getData();
                    double total = 0;
                    @Nullable final double[] ld = l.getData();
                    @Nullable final double[] rd = r.getData();
                    if (null != r)
                      r.freeRef();
                    for (int i = 0; i < l.length(); i++) {
                      final double lv = Math.max(Math.min(ld[i], max_prob), zero_tol);
                      final double rv = rd[i];
                      if (rv > 0) {
                        gradientData[i] = -rv / lv;
                        total += -rv * Math.log(lv);
                      } else {
                        gradientData[i] = 0;
                      }
                    }
                    if (null != l)
                      l.freeRef();
                    assert total >= 0;
                    {
                      Tensor temp_03_0001 = gradientTensor == null ? null
                          : gradientTensor.addRef();
                      if (null != gradient[dataIndex])
                        gradient[dataIndex].freeRef();
                      gradient[dataIndex] = temp_03_0001 == null ? null : temp_03_0001.addRef();
                      if (null != temp_03_0001)
                        temp_03_0001.freeRef();
                    }
                    gradientTensor.freeRef();
                    return new Tensor(new double[]{total}, 1);
                  }, indata == null ? null : indata.addRef(), Result.addRefs(inObj),
                  Tensor.addRefs(gradient))).toArray(i -> new Tensor[i])),
              new Result.Accumulator() {
                {
                  Result.addRefs(inObj);
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (inObj[1].isAlive()) {
                    @Nonnull
                    TensorArray tensorArray = new TensorArray(
                        RefIntStream.range(0, delta.length()).mapToObj(RefUtil.wrapInterface(
                            (IntFunction<? extends Tensor>) dataIndex -> {
                              Tensor deltaTensor = delta.get(dataIndex);
                              @Nullable final Tensor inputTensor = indata.get(dataIndex);
                              @Nonnull final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
                              for (int i = 0; i < passback.length(); i++) {
                                final double lv = Math.max(Math.min(inputTensor.get(i), max_prob), zero_tol);
                                RefUtil
                                    .freeRef(passback.set(i, -deltaTensor.get(0) * Math.log(lv)));
                              }
                              if (null != inputTensor)
                                inputTensor.freeRef();
                              if (null != deltaTensor)
                                deltaTensor.freeRef();
                              return passback;
                            }, indata == null ? null : indata.addRef(),
                            Tensor.addRefs(gradient),
                            delta == null ? null : delta.addRef())).toArray(i -> new Tensor[i]));
                    inObj[1].accumulate(buffer == null ? null : buffer.addRef(),
                        tensorArray == null ? null : tensorArray);
                  }
                  if (inObj[0].isAlive()) {
                    @Nonnull
                    TensorArray tensorArray = new TensorArray(
                        RefIntStream.range(0, delta.length()).mapToObj(RefUtil.wrapInterface(
                            (IntFunction<? extends Tensor>) dataIndex -> {
                              Tensor tensor = delta.get(dataIndex);
                              @Nonnull final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
                              for (int i = 0; i < passback.length(); i++) {
                                RefUtil
                                    .freeRef(passback.set(i, tensor.get(0) * gradient[dataIndex].get(i)));
                              }
                              if (null != tensor)
                                tensor.freeRef();
                              return passback;
                            }, delta == null ? null : delta.addRef(),
                            Tensor.addRefs(gradient))).toArray(i -> new Tensor[i]));
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
                  ReferenceCounting.freeRefs(inObj);
                }
              }) {

            {
              Result.addRefs(inObj);
            }

            @Override
            public boolean isAlive() {
              return inObj[0].isAlive() || inObj[0].isAlive();
            }

            public void _free() {
              ReferenceCounting.freeRefs(inObj);
            }

          };
        } finally {
          ReferenceCounting.freeRefs(inObj);
        }
      } finally {
        ReferenceCounting.freeRefs(gradient);
      }
    } finally {
      if (null != indata)
        indata.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  EntropyLossLayer addRef() {
    return (EntropyLossLayer) super.addRef();
  }
}
