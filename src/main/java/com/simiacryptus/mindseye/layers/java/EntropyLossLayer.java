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
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class EntropyLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  public EntropyLossLayer() {
  }

  protected EntropyLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static EntropyLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new EntropyLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    final double zero_tol = 1e-12;
    TensorList indata = inObj[0].getData();
    @Nonnull final RefArrayList<Tensor> gradient = new RefArrayList<>();
    final double max_prob = 1.;
    try {
      TensorArray data = new TensorArray(RefIntStream.range(0, indata.length())
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            @Nullable final Tensor l = indata.get(dataIndex);
            TensorList temp_03_0006 = inObj[1].getData();
            @Nullable final Tensor r = temp_03_0006.get(dataIndex);
            temp_03_0006.freeRef();
            if (l.length() != r.length()) {
              IllegalArgumentException temp_03_0004 = new IllegalArgumentException(
                  l.length() + " != " + r.length());
              l.freeRef();
              r.freeRef();
              throw temp_03_0004;
            }
            @Nonnull final Tensor gradientTensor = new Tensor(l.getDimensions());
            @Nullable final double[] gradientData = gradientTensor.getData();
            double total = 0;
            @Nullable final double[] ld = l.getData();
            @Nullable final double[] rd = r.getData();
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
            l.freeRef();
            assert total >= 0;
            gradient.add(dataIndex, gradientTensor);
            //RefUtil.set(gradient, dataIndex, gradientTensor);
            return new Tensor(new double[]{total}, 1);
          }, indata.addRef(), RefUtil.addRefs(inObj), RefUtil.addRef(gradient)))
          .toArray(i -> new Tensor[i]));
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
          indata.addRef();
          RefUtil.addRef(gradient);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (inObj[1].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                      Tensor deltaTensor = delta.get(dataIndex);
                      @Nullable final Tensor inputTensor = indata.get(dataIndex);
                      Tensor tensor = gradient.get(dataIndex);
                      @Nonnull final Tensor passback = new Tensor(tensor.getDimensions());
                      tensor.freeRef();
                      for (int i = 0; i < passback.length(); i++) {
                        final double lv = Math.max(Math.min(inputTensor.get(i), max_prob), zero_tol);
                        final double value = -deltaTensor.get(0) * Math.log(lv);
                        passback.set(i, value);
                      }
                      inputTensor.freeRef();
                      deltaTensor.freeRef();
                      return passback;
                    }, indata.addRef(), RefUtil.addRef(gradient),
                    delta.addRef()))
                .toArray(i -> new Tensor[i]));
            inObj[1].accumulate(buffer == null ? null : buffer.addRef(),
                tensorArray);
          }
          if (inObj[0].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor tensor = delta.get(dataIndex);
                  Tensor tensor1 = gradient.get(dataIndex);
                  @Nonnull final Tensor passback = new Tensor(tensor1.getDimensions());
                  for (int i = 0; i < passback.length(); i++) {
                    passback.set(i, tensor.get(0) * tensor1.get(i));
                  }
                  tensor1.freeRef();
                  tensor.freeRef();
                  return passback;
                }, delta.addRef(), RefUtil.addRef(gradient)))
                .toArray(i -> new Tensor[i]));
            inObj[0].accumulate(buffer == null ? null : buffer.addRef(),
                tensorArray);
          }
          delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRefs(inObj);
          indata.freeRef();
          RefUtil.freeRef(gradient);
        }
      };
      return new Result(data, accumulator) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          if (inObj[0].isAlive()) return true;
          inObj[0].isAlive();
          return false;
        }

        public void _free() {
          RefUtil.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRefs(inObj);
      RefUtil.freeRef(gradient);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  EntropyLossLayer addRef() {
    return (EntropyLossLayer) super.addRef();
  }
}
