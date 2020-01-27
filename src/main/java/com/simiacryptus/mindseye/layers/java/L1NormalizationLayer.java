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
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class L1NormalizationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  double maxInput = 50;

  public L1NormalizationLayer() {
  }

  protected L1NormalizationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static L1NormalizationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new L1NormalizationLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... input) {
    final Result in = input[0].addRef();
    RefUtil.freeRefs(input);
    final TensorList inData = in.getData();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          in.addRef();
          inData.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList outDelta) {
          if (in.isAlive()) {
            final Tensor[] passbackArray = RefIntStream.range(0, outDelta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor inputTensor = inData.get(dataIndex);
                  @Nullable final double[] value = inputTensor.getData();
                  inputTensor.freeRef();
                  Tensor outputTensor = outDelta.get(dataIndex);
                  @Nullable final double[] delta = outputTensor.getData();
                  final double dot = ArrayUtil.dot(value, delta);
                  final double sum = RefArrays.stream(value).sum();
                  @Nonnull final Tensor passback = new Tensor(outputTensor.getDimensions());
                  outputTensor.freeRef();
                  @Nullable final double[] passbackData = passback.getData();
                  if (0 != sum || Double.isFinite(sum)) {
                    for (int i = 0; i < value.length; i++) {
                      passbackData[i] = (delta[i] - dot / sum) / sum;
                    }
                  }
                  return passback;
                }, inData.addRef(), outDelta.addRef()))
                .toArray(i -> new Tensor[i]);
            assert RefArrays.stream(RefUtil.addRefs(passbackArray)).flatMapToDouble(x -> {
              RefDoubleStream temp_26_0004 = RefArrays.stream(x.getData());
              x.freeRef();
              return temp_26_0004;
            }).allMatch(v -> Double.isFinite(v));
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefUtil.addRefs(passbackArray));
            RefUtil.freeRefs(passbackArray);
            in.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          outDelta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          in.freeRef();
          inData.freeRef();
        }
      };
      TensorArray data = new TensorArray(RefIntStream.range(0, inData.length())
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            @Nullable final Tensor value = inData.get(dataIndex);
            final double sum = value.sum();
            if (!Double.isFinite(sum) || 0 == sum) {
              return value;
            } else {
              Tensor temp_26_0003 = value.scale(1.0 / sum);
              value.freeRef();
              return temp_26_0003;
            }
          }, inData.addRef())).toArray(i -> new Tensor[i]));
      return new Result(data, accumulator) {
        {
          in.addRef();
        }
        @Override
        public boolean isAlive() {
          return in.isAlive();
        }

        @Override
        public void _free() {
          in.freeRef();
          super._free();
        }
      };
    } finally {
      inData.freeRef();
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
  L1NormalizationLayer addRef() {
    return (L1NormalizationLayer) super.addRef();
  }
}
