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
public @com.simiacryptus.ref.lang.RefAware class CrossDifferenceLayer extends LayerBase {

  public CrossDifferenceLayer() {
  }

  protected CrossDifferenceLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static CrossDifferenceLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new CrossDifferenceLayer(json);
  }

  public static int index(final int x, final int y, final int max) {
    return max * (max - 1) / 2 - (max - x) * (max - x - 1) / 2 + y - x - 1;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    return new Result(new TensorArray(inObj[0].getData().stream().parallel().map(tensor -> {
      final int inputDim = tensor.length();
      final int outputDim = (inputDim * inputDim - inputDim) / 2;
      @Nonnull
      final Tensor result1 = new Tensor(outputDim);
      @Nullable
      final double[] inputData = tensor.getData();
      @Nullable
      final double[] resultData = result1.getData();
      com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputDim).forEach(x -> {
        com.simiacryptus.ref.wrappers.RefIntStream.range(x + 1, inputDim).forEach(y -> {
          resultData[CrossDifferenceLayer.index(x, y, inputDim)] = inputData[x] - inputData[y];
        });
      });
      return result1;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      final Result input = inObj[0];
      if (input.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(data.stream().parallel().map(tensor -> {
          final int outputDim = tensor.length();
          final int inputDim = (1 + (int) Math.sqrt(1 + 8 * outputDim)) / 2;
          @Nonnull
          final Tensor passback = new Tensor(inputDim);
          @Nullable
          final double[] passbackData = passback.getData();
          @Nullable
          final double[] tensorData = tensor.getData();
          com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputDim).forEach(x -> {
            com.simiacryptus.ref.wrappers.RefIntStream.range(x + 1, inputDim).forEach(y -> {
              passbackData[x] += tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
              passbackData[y] += -tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
            });
          });
          return passback;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull
        final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
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

  public @Override @SuppressWarnings("unused") CrossDifferenceLayer addRef() {
    return (CrossDifferenceLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") CrossDifferenceLayer[] addRefs(CrossDifferenceLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CrossDifferenceLayer::addRef)
        .toArray((x) -> new CrossDifferenceLayer[x]);
  }

  public static @SuppressWarnings("unused") CrossDifferenceLayer[][] addRefs(CrossDifferenceLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(CrossDifferenceLayer::addRefs)
        .toArray((x) -> new CrossDifferenceLayer[x][]);
  }

}
