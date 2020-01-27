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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class CrossProductLayer extends LayerBase {

  public CrossProductLayer() {
  }

  protected CrossProductLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static CrossProductLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CrossProductLayer(json);
  }

  public static int index(final int x, final int y, final int max) {
    return max * (max - 1) / 2 - (max - x) * (max - x - 1) / 2 + y - x - 1;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    final Result in = inObj[0].addRef();
    TensorList indata = in.getData();
    try {
      try {
        return new Result(new TensorArray(indata.stream().parallel().map(tensor -> {
          final int inputDim = tensor.length();
          final int outputDim = (inputDim * inputDim - inputDim) / 2;
          @Nonnull final Tensor result1 = new Tensor(outputDim);
          @Nullable final double[] inputData = tensor.getData();
          tensor.freeRef();
          @Nullable final double[] resultData = result1.getData();
          RefIntStream.range(0, inputDim).forEach(x -> {
            RefIntStream.range(x + 1, inputDim).forEach(y -> {
              resultData[CrossProductLayer.index(x, y, inputDim)] = inputData[x] * inputData[y];
            });
          });
          return result1;
        }).toArray(i -> new Tensor[i])), new Result.Accumulator() {
          {
            in.addRef();
            indata.addRef();
          }

          @Override
          public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
            if (in.isAlive()) {
              assert delta.length() == delta.length();
              @Nonnull
              TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length()).parallel()
                  .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) batchIndex -> {
                    @Nullable final Tensor deltaTensor = delta.get(batchIndex);
                    final int outputDim = deltaTensor.length();
                    final int inputDim = (1 + (int) Math.sqrt(1 + 8 * outputDim)) / 2;
                    @Nonnull final Tensor passback = new Tensor(inputDim);
                    @Nullable final double[] passbackData = passback.getData();
                    @Nullable final double[] tensorData = deltaTensor.getData();
                    deltaTensor.freeRef();
                    Tensor inputTensor = indata.get(batchIndex);
                    @Nullable final double[] inputData = inputTensor.getData();
                    inputTensor.freeRef();
                    RefIntStream.range(0, inputDim).forEach(x -> {
                      RefIntStream.range(x + 1, inputDim).forEach(y -> {
                        passbackData[x] += tensorData[CrossProductLayer.index(x, y, inputDim)] * inputData[y];
                        passbackData[y] += tensorData[CrossProductLayer.index(x, y, inputDim)] * inputData[x];
                      });
                    });
                    return passback;
                  }, indata.addRef(), delta.addRef()))
                  .toArray(i -> new Tensor[i]));
              in.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
            }
            delta.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
            super._free();
            in.freeRef();
            indata.freeRef();
          }
        }) {

          {
            RefUtil.addRefs(inObj);
          }

          @Override
          public boolean isAlive() {
            for (@Nonnull final Result element : inObj)
              if (element.isAlive()) {
                return true;
              }
            return false;
          }

          public void _free() {
            RefUtil.freeRefs(inObj);
            super._free();
          }
        };
      } finally {
        RefUtil.freeRefs(inObj);
        indata.freeRef();
      }
    } finally {
      in.freeRef();
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
  CrossProductLayer addRef() {
    return (CrossProductLayer) super.addRef();
  }

}
