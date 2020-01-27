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
import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.*;

@SuppressWarnings("serial")
public class MaxPoolingLayer extends LayerBase {

  private static final Function<MaxPoolingLayer.CalcRegionsParameter, RefList<Tuple2<Integer, int[]>>> calcRegionsCache = Util
      .cache(MaxPoolingLayer::calcRegions);
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxPoolingLayer.class);
  private int[] kernelDims;

  protected MaxPoolingLayer() {
    super();
  }

  public MaxPoolingLayer(@Nonnull final int... kernelDims) {

    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  protected MaxPoolingLayer(@Nonnull final JsonObject id, @Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MaxPoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  private static RefList<Tuple2<Integer, int[]>> calcRegions(@Nonnull final MaxPoolingLayer.CalcRegionsParameter p) {
    @Nonnull final Tensor input = new Tensor(p.inputDims);
    final int[] newDims = RefIntStream.range(0, p.inputDims.length).map(i -> {
      //assert 0 == p.inputDims[i] % p.kernelDims[i];
      return (int) Math.ceil(p.inputDims[i] * 1.0 / p.kernelDims[i]);
    }).toArray();
    @Nonnull final Tensor output = new Tensor(newDims);

    RefList<Tuple2<Integer, int[]>> temp_53_0001 = output.coordStream(true)
        .map(RefUtil.wrapInterface((Function<? super Coordinate, ? extends Tuple2<Integer, int[]>>) o -> {
          Tensor tensor = new Tensor(p.kernelDims);
          final int[] inCoords = tensor.coordStream(true)
              .mapToInt(RefUtil.wrapInterface((ToIntFunction<? super Coordinate>) kernelCoord -> {
                @Nonnull final int[] result = new int[o.getCoords().length];
                for (int index = 0; index < o.getCoords().length; index++) {
                  final int outputCoordinate = o.getCoords()[index];
                  final int kernelSize = p.kernelDims[index];
                  final int baseCoordinate = Math.min(outputCoordinate * kernelSize, p.inputDims[index] - kernelSize);
                  final int kernelCoordinate = kernelCoord.getCoords()[index];
                  result[index] = baseCoordinate + kernelCoordinate;
                }
                return input.index(result);
              }, input.addRef())).toArray();
          tensor.freeRef();
          return new Tuple2<>(o.getIndex(), inCoords);
        }, input)).collect(RefCollectors.toList());
    output.freeRef();
    return temp_53_0001;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    final Result in = inObj[0].addRef();
    RefUtil.freeRefs(inObj);
    TensorList temp_53_0005 = in.getData();
    temp_53_0005.length();

    temp_53_0005.freeRef();
    TensorList temp_53_0006 = in.getData();
    @Nonnull final int[] inputDims = temp_53_0006.getDimensions();
    temp_53_0006.freeRef();
    final RefList<Tuple2<Integer, int[]>> regions = MaxPoolingLayer.calcRegionsCache
        .apply(new MaxPoolingLayer.CalcRegionsParameter(inputDims, kernelDims));
    TensorList temp_53_0007 = in.getData();
    final Tensor[] outputA = RefIntStream.range(0, temp_53_0007.length()).mapToObj(dataIndex -> {
      final int[] newDims = RefIntStream.range(0, inputDims.length).map(i -> {
        return (int) Math.ceil(inputDims[i] * 1.0 / kernelDims[i]);
      }).toArray();
      return new Tensor(newDims);
    }).toArray(i -> new Tensor[i]);
    temp_53_0007.freeRef();
    RefArrays.stream(RefUtil.addRefs(outputA)).mapToInt(x -> {
      int temp_53_0004 = x.length();
      x.freeRef();
      return temp_53_0004;
    }).sum();
    TensorList temp_53_0008 = in.getData();
    @Nonnull final int[][] gradientMapA = new int[temp_53_0008.length()][];
    temp_53_0008.freeRef();
    TensorList temp_53_0009 = in.getData();
    RefIntStream.range(0, temp_53_0009.length()).forEach(RefUtil.wrapInterface(dataIndex -> {
      TensorList temp_53_0010 = in.getData();
      @Nullable final Tensor input = temp_53_0010.get(dataIndex);
      temp_53_0010.freeRef();
      final Tensor output = outputA[dataIndex].addRef();
      @Nonnull final IntToDoubleFunction keyExtractor = RefUtil.wrapInterface(inputCoords -> input.get(inputCoords),
          input.addRef());
      @Nonnull final int[] gradientMap = new int[input.length()];
      regions.parallelStream().forEach(RefUtil.wrapInterface((Consumer<? super Tuple2<Integer, int[]>>) tuple -> {
        final Integer from = tuple.getFirst();
        final int[] toList = tuple.getSecond();
        int toMax = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (final int c : toList) {
          final double value = keyExtractor.applyAsDouble(c);
          if (-1 == toMax || bestValue < value) {
            bestValue = value;
            toMax = c;
          }
        }
        gradientMap[from] = toMax;
        output.set(from, input.get(toMax));
      }, output.addRef(), input.addRef()));
      output.freeRef();
      input.freeRef();
      gradientMapA[dataIndex] = gradientMap;
    }, in.addRef(), RefUtil.addRefs(outputA), regions == null ? null : regions.addRef()));
    temp_53_0009.freeRef();
    if (null != regions)
      regions.freeRef();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          in.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (in.isAlive()) {
            TensorList temp_53_0011 = in.getData();
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, temp_53_0011.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nonnull final Tensor backSignal = new Tensor(inputDims);
                  final int[] ints = gradientMapA[dataIndex];
                  @Nullable final Tensor datum = data.get(dataIndex);
                  for (int i = 0; i < datum.length(); i++) {
                    backSignal.add(ints[i], datum.get(i));
                  }
                  datum.freeRef();
                  return backSignal;
                }, data.addRef())).toArray(i -> new Tensor[i]));
            temp_53_0011.freeRef();
            in.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          in.freeRef();
        }
      };
      return new Result(new TensorArray(RefUtil.addRefs(outputA)), accumulator) {
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
      RefUtil.freeRefs(outputA);
      in.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("heapCopy", JsonUtil.getJson(kernelDims));
    return json;
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
  MaxPoolingLayer addRef() {
    return (MaxPoolingLayer) super.addRef();
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
      @Nonnull final MaxPoolingLayer.CalcRegionsParameter other = (MaxPoolingLayer.CalcRegionsParameter) obj;
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
