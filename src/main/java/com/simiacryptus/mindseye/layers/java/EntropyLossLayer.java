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
public @com.simiacryptus.ref.lang.RefAware class EntropyLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  public EntropyLossLayer() {
  }

  protected EntropyLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static EntropyLossLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new EntropyLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final double zero_tol = 1e-12;
    TensorList indata = inObj[0].getData();
    @Nonnull
    final Tensor gradient[] = new Tensor[indata.length()];
    final double max_prob = 1.;
    return new Result(
        new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, indata.length()).mapToObj(dataIndex -> {
          @Nullable
          final Tensor l = indata.get(dataIndex);
          @Nullable
          final Tensor r = inObj[1].getData().get(dataIndex);
          if (l.length() != r.length()) {
            throw new IllegalArgumentException(l.length() + " != " + r.length());
          }
          @Nonnull
          final Tensor gradientTensor = new Tensor(l.getDimensions());
          @Nullable
          final double[] gradientData = gradientTensor.getData();
          double total = 0;
          @Nullable
          final double[] ld = l.getData();
          @Nullable
          final double[] rd = r.getData();
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
          assert total >= 0;
          gradient[dataIndex] = gradientTensor;
          return new Tensor(new double[] { total }, 1);
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (inObj[1].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nullable
                  final Tensor inputTensor = indata.get(dataIndex);
                  @Nonnull
                  final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
                  for (int i = 0; i < passback.length(); i++) {
                    final double lv = Math.max(Math.min(inputTensor.get(i), max_prob), zero_tol);
                    passback.set(i, -deltaTensor.get(0) * Math.log(lv));
                  }
                  return passback;
                }).toArray(i -> new Tensor[i]));
            inObj[1].accumulate(buffer, tensorArray);
          }
          if (inObj[0].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
                  Tensor tensor = delta.get(dataIndex);
                  @Nonnull
                  final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
                  for (int i = 0; i < passback.length(); i++) {
                    passback.set(i, tensor.get(0) * gradient[dataIndex].get(i));
                  }
                  return passback;
                }).toArray(i -> new Tensor[i]));
            inObj[0].accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[0].isAlive();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") EntropyLossLayer addRef() {
    return (EntropyLossLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") EntropyLossLayer[] addRefs(EntropyLossLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(EntropyLossLayer::addRef)
        .toArray((x) -> new EntropyLossLayer[x]);
  }

  public static @SuppressWarnings("unused") EntropyLossLayer[][] addRefs(EntropyLossLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(EntropyLossLayer::addRefs)
        .toArray((x) -> new EntropyLossLayer[x][]);
  }
}
