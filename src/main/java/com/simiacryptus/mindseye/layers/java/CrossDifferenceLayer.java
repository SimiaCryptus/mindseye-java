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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class CrossDifferenceLayer extends LayerBase {

  public CrossDifferenceLayer() {
  }

  protected CrossDifferenceLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static CrossDifferenceLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CrossDifferenceLayer(json);
  }

  public static int index(final int x, final int y, final int max) {
    return max * (max - 1) / 2 - (max - x) * (max - x - 1) / 2 + y - x - 1;
  }

  public static @SuppressWarnings("unused") CrossDifferenceLayer[] addRefs(CrossDifferenceLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CrossDifferenceLayer::addRef)
        .toArray((x) -> new CrossDifferenceLayer[x]);
  }

  public static @SuppressWarnings("unused") CrossDifferenceLayer[][] addRefs(CrossDifferenceLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CrossDifferenceLayer::addRefs)
        .toArray((x) -> new CrossDifferenceLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    try {
      TensorList temp_65_0003 = inObj[0].getData();
      Result temp_65_0002 = new Result(new TensorArray(temp_65_0003.stream().parallel().map(tensor -> {
        final int inputDim = tensor.length();
        final int outputDim = (inputDim * inputDim - inputDim) / 2;
        @Nonnull
        final Tensor result1 = new Tensor(outputDim);
        @Nullable
        final double[] inputData = tensor.getData();
        if (null != tensor)
          tensor.freeRef();
        @Nullable
        final double[] resultData = result1.getData();
        RefIntStream.range(0, inputDim).forEach(x -> {
          RefIntStream.range(x + 1, inputDim).forEach(y -> {
            resultData[CrossDifferenceLayer.index(x, y, inputDim)] = inputData[x] - inputData[y];
          });
        });
        return result1;
      }).toArray(i -> new Tensor[i])), new Result.Accumulator() {
        {
          Result.addRefs(inObj);
        }

        @Override
        public void accept(DeltaSet<UUID> buffer, TensorList data) {
          final Result input = inObj[0].addRef();
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
              if (null != tensor)
                tensor.freeRef();
              RefIntStream.range(0, inputDim).forEach(x -> {
                RefIntStream.range(x + 1, inputDim).forEach(y -> {
                  passbackData[x] += tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
                  passbackData[y] += -tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
                });
              });
              return passback;
            }).toArray(i -> new Tensor[i]));
            input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
          }
          if (null != data)
            data.freeRef();
          if (null != buffer)
            buffer.freeRef();
          if (null != input)
            input.freeRef();
        }

        public @SuppressWarnings("unused") void _free() {
          ReferenceCounting.freeRefs(inObj);
        }
      }) {

        {
          Result.addRefs(inObj);
        }

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
          ReferenceCounting.freeRefs(inObj);
        }

      };
      if (null != temp_65_0003)
        temp_65_0003.freeRef();
      return temp_65_0002;
    } finally {
      ReferenceCounting.freeRefs(inObj);
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") CrossDifferenceLayer addRef() {
    return (CrossDifferenceLayer) super.addRef();
  }

}
