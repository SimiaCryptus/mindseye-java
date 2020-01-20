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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class MaxImageBandLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxImageBandLayer.class);

  public MaxImageBandLayer() {
    super();
  }

  protected MaxImageBandLayer(@Nonnull final JsonObject id, final int... kernelDims) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MaxImageBandLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxImageBandLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
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
      Coordinate[] temp_31_0002 = RefIntStream.range(0, inputDims[2])
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Coordinate>) band -> {
            return RefUtil.get(data.coordStream(true).filter(e -> e.getCoords()[2] == band)
                .max(RefComparator
                    .comparing(RefUtil.wrapInterface((Function<? super Coordinate, ? extends Double>) c -> data.get(c),
                        data.addRef()))));
          }, data == null ? null : data.addRef())).toArray(i -> new Coordinate[i]);
      if (null != data)
        data.freeRef();
      return temp_31_0002;
    }).toArray(i -> new Coordinate[i][]);

    try {
      return new Result(new TensorArray(RefIntStream.range(0, inputData.length())
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            Tensor tensor = inputData.get(dataIndex);
            final RefDoubleStream doubleStream = RefIntStream.range(0, inputDims[2])
                .mapToDouble(RefUtil.wrapInterface((IntToDoubleFunction) band -> {
                  final int[] maxCoord = maxCoords[dataIndex][band].getCoords();
                  return tensor.get(maxCoord[0], maxCoord[1], band);
                }, tensor.addRef()));
            tensor.freeRef();
            Tensor temp_31_0005 = new Tensor(1, 1, inputDims[2]);
            temp_31_0005.set(Tensor.getDoubles(doubleStream, inputDims[2]));
            Tensor temp_31_0004 = temp_31_0005.addRef();
            temp_31_0005.freeRef();
            return temp_31_0004;
          }, inputData.addRef())).toArray(i -> new Tensor[i])), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
          inputData.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (inObj[0].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nonnull final Tensor passback = new Tensor(inputData.getDimensions());
                  RefIntStream.range(0, inputDims[2]).forEach(RefUtil.wrapInterface(b -> {
                        final int[] maxCoord = maxCoords[dataIndex][b].getCoords();
                        passback.set(new int[]{maxCoord[0], maxCoord[1], b}, deltaTensor.get(0, 0, b));
                      }, passback.addRef(),
                      deltaTensor.addRef()));
                  deltaTensor.freeRef();
                  return passback;
                }, delta.addRef(), inputData.addRef()))
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
          ReferenceCounting.freeRefs(inObj);
          inputData.freeRef();
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return inObj[0].isAlive();
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
          super._free();
        }
      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
      inputData.freeRef();
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MaxImageBandLayer addRef() {
    return (MaxImageBandLayer) super.addRef();
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
      @Nonnull final MaxImageBandLayer.CalcRegionsParameter other = (MaxImageBandLayer.CalcRegionsParameter) obj;
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
}
