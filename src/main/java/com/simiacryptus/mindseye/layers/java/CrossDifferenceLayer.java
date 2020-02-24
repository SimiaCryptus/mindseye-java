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
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class CrossDifferenceLayer extends LayerBase {

  public CrossDifferenceLayer() {
  }

  protected CrossDifferenceLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static CrossDifferenceLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CrossDifferenceLayer(json);
  }

  public static int index(final int x, final int y, final int max) {
    return max * (max - 1) / 2 - (max - x) * (max - x - 1) / 2 + y - x - 1;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    TensorArray data = fwd(inObj[0].getData());
    Accumulator accumulator = new Accumulator(inObj[0].getAccumulator(), inObj[0].isAlive());
    boolean alive = alive(inObj);
    return new Result(data, accumulator, alive);
  }

  private boolean alive(Result[] inObj) {
    return Result.anyAlive(inObj);
  }

  @NotNull
  private TensorArray fwd(TensorList input) {
    TensorArray tensorArray = new TensorArray(input.stream().parallel().map(tensor -> {
      final int inputDim = tensor.length();
      final int outputDim = (inputDim * inputDim - inputDim) / 2;
      @Nonnull final Tensor result1 = new Tensor(outputDim);
      @Nullable final double[] inputData = tensor.getData();
      tensor.freeRef();
      @Nullable final double[] resultData = result1.getData();
      RefIntStream.range(0, inputDim).forEach(x -> {
        RefIntStream.range(x + 1, inputDim).forEach(y -> {
          resultData[CrossDifferenceLayer.index(x, y, inputDim)] = inputData[x] - inputData[y];
        });
      });
      return result1;
    }).toArray(Tensor[]::new));
    input.freeRef();
    return tensorArray;
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
  CrossDifferenceLayer addRef() {
    return (CrossDifferenceLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private boolean alive;
    private Result.Accumulator accumulator;

    public Accumulator(Result.Accumulator accumulator, boolean alive) {
      this.alive = alive;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(data.stream().parallel().map(tensor -> {
          final int outputDim = tensor.length();
          final int inputDim = (1 + (int) Math.sqrt(1 + 8 * outputDim)) / 2;
          @Nonnull final Tensor passback = new Tensor(inputDim);
          @Nullable final double[] passbackData = passback.getData();
          @Nullable final double[] tensorData = tensor.getData();
          tensor.freeRef();
          RefIntStream.range(0, inputDim).forEach(x -> {
            RefIntStream.range(x + 1, inputDim).forEach(y -> {
              passbackData[x] += tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
              passbackData[y] += -tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
            });
          });
          return passback;
        }).toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      data.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
