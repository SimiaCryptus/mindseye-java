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
import com.simiacryptus.ref.wrappers.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class MaxImageBandLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxImageBandLayer.class);

  public MaxImageBandLayer() {
    super();
  }

  protected MaxImageBandLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MaxImageBandLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxImageBandLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    assert 1 == inObj.length;
    final TensorList inputData = inObj[0].getData();
    inputData.length();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    final Coordinate[][] maxCoords = inputData.stream().map(data -> {
      return RefIntStream.range(0, inputDims[2])
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Coordinate>) band -> {
            return RefUtil.get(data.coordStream(true).filter(e -> e.getCoords()[2] == band)
                .max(RefComparator
                    .comparingDouble(RefUtil.wrapInterface(data::get,
                        data.addRef()))));
          }, data)).toArray(Coordinate[]::new);
    }).toArray(Coordinate[][]::new);

    TensorArray data = fwd(inputData.addRef(), inputDims[2], maxCoords);
    boolean alive = inObj[0].isAlive();
    final Result.Accumulator accumulator1 = inObj[0].getAccumulator();
    final boolean alive1 = inObj[0].isAlive();
    Accumulator accumulator = new Accumulator(inputData, inputDims, maxCoords, accumulator1, alive1);
    RefUtil.freeRef(inObj);
    return new Result(data, accumulator, alive);
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
  MaxImageBandLayer addRef() {
    return (MaxImageBandLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inputData, int inputDim, Coordinate[][] maxCoords) {
    return new TensorArray(RefIntStream.range(0, inputData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor inTensor = inputData.get(dataIndex);
          final RefDoubleStream doubleStream = RefIntStream.range(0, inputDim)
              .mapToDouble(RefUtil.wrapInterface((IntToDoubleFunction) band -> {
                final int[] maxCoord = maxCoords[dataIndex][band].getCoords();
                return inTensor.get(maxCoord[0], maxCoord[1], band);
              }, inTensor));
          Tensor outTensor = new Tensor(1, 1, inputDim);
          outTensor.set(Tensor.getDoubles(doubleStream, inputDim));
          return outTensor;
        }, inputData)).toArray(Tensor[]::new));
  }

  public static class CalcRegionsParameter {
    public final int[] inputDims;
    public final int[] kernelDims;

    public CalcRegionsParameter(final int[] inputDims, final int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }

    @Override
    public boolean equals(@Nullable final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      final MaxImageBandLayer.CalcRegionsParameter other = (MaxImageBandLayer.CalcRegionsParameter) obj;
      if (!RefArrays.equals(inputDims, other.inputDims)) {
        return false;
      }
      return RefArrays.equals(kernelDims, other.kernelDims);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + RefArrays.hashCode(inputDims);
      result = prime * result + RefArrays.hashCode(kernelDims);
      return result;
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList inputData;
    private final int[] inputDims;
    private final Coordinate[][] maxCoords;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(TensorList inputData, int[] inputDims, Coordinate[][] maxCoords, Result.Accumulator accumulator, boolean alive) {
      this.inputData = inputData;
      this.inputDims = inputDims;
      this.maxCoords = maxCoords;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              Tensor deltaTensor = delta.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(inputData.getDimensions());
              RefIntStream.range(0, inputDims[2]).forEach(RefUtil.wrapInterface(b -> {
                    final int[] maxCoord = maxCoords[dataIndex][b].getCoords();
                    passback.set(new int[]{maxCoord[0], maxCoord[1], b}, deltaTensor.get(0, 0, b));
                  }, passback.addRef(),
                  deltaTensor));
              return passback;
            }, delta.addRef(), inputData.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        try {
          this.accumulator.accept(buffer1, tensorArray);
        } finally {
          this.accumulator.freeRef();
        }
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(accumulator);
      inputData.freeRef();
    }
  }
}
