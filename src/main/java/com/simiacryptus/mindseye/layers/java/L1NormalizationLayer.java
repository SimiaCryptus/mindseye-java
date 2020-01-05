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
import com.simiacryptus.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class L1NormalizationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  double maxInput = 50;

  public L1NormalizationLayer() {
  }

  protected L1NormalizationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static L1NormalizationLayer fromJson(@Nonnull final JsonObject json,
                                              Map<CharSequence, byte[]> rs) {
    return new L1NormalizationLayer(json);
  }

  public static @SuppressWarnings("unused")
  L1NormalizationLayer[] addRefs(L1NormalizationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(L1NormalizationLayer::addRef)
        .toArray((x) -> new L1NormalizationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  L1NormalizationLayer[][] addRefs(L1NormalizationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(L1NormalizationLayer::addRefs)
        .toArray((x) -> new L1NormalizationLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... input) {
    final Result in = input[0];
    final TensorList inData = in.getData();
    return new Result(
        new TensorArray(RefIntStream.range(0, inData.length()).mapToObj(dataIndex -> {
          @Nullable final Tensor value = inData.get(dataIndex);
          {
            final double sum = value.sum();
            if (!Double.isFinite(sum) || 0 == sum) {
              return value;
            } else {
              return value.scale(1.0 / sum);
            }
          }
        }).toArray(i -> new Tensor[i])), new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList outDelta) {
        if (in.isAlive()) {
          final Tensor[] passbackArray = RefIntStream.range(0, outDelta.length())
              .mapToObj(dataIndex -> {
                Tensor inputTensor = inData.get(dataIndex);
                @Nullable final double[] value = inputTensor.getData();
                Tensor outputTensor = outDelta.get(dataIndex);
                @Nullable final double[] delta = outputTensor.getData();
                final double dot = ArrayUtil.dot(value, delta);
                final double sum = RefArrays.stream(value).sum();
                @Nonnull final Tensor passback = new Tensor(outputTensor.getDimensions());
                @Nullable final double[] passbackData = passback.getData();
                if (0 != sum || Double.isFinite(sum)) {
                  for (int i = 0; i < value.length; i++) {
                    passbackData[i] = (delta[i] - dot / sum) / sum;
                  }
                }
                return passback;
              }).toArray(i -> new Tensor[i]);
          assert RefArrays.stream(passbackArray)
              .flatMapToDouble(x -> RefArrays.stream(x.getData()))
              .allMatch(v -> Double.isFinite(v));
          @Nonnull
          TensorArray tensorArray = new TensorArray(passbackArray);
          in.accumulate(buffer, tensorArray);
        }
      }
    }) {

      @Override
      public boolean isAlive() {
        return in.isAlive();
      }

      public void _free() {

      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
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
  L1NormalizationLayer addRef() {
    return (L1NormalizationLayer) super.addRef();
  }
}
