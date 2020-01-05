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
import com.simiacryptus.ref.lang.RefAware;
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
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public @RefAware
class MaxPoolingLayer extends LayerBase {

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

  @SuppressWarnings("unused")
  public static MaxPoolingLayer fromJson(@Nonnull final JsonObject json,
                                         Map<CharSequence, byte[]> rs) {
    return new MaxPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  public static @SuppressWarnings("unused")
  MaxPoolingLayer[] addRefs(MaxPoolingLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxPoolingLayer::addRef)
        .toArray((x) -> new MaxPoolingLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MaxPoolingLayer[][] addRefs(MaxPoolingLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxPoolingLayer::addRefs)
        .toArray((x) -> new MaxPoolingLayer[x][]);
  }

  private static RefList<Tuple2<Integer, int[]>> calcRegions(
      @Nonnull final MaxPoolingLayer.CalcRegionsParameter p) {
    @Nonnull final Tensor input = new Tensor(p.inputDims);
    final int[] newDims = RefIntStream.range(0, p.inputDims.length).map(i -> {
      //assert 0 == p.inputDims[i] % p.kernelDims[i];
      return (int) Math.ceil(p.inputDims[i] * 1.0 / p.kernelDims[i]);
    }).toArray();
    @Nonnull final Tensor output = new Tensor(newDims);

    return output.coordStream(true).map(o -> {
      Tensor tensor = new Tensor(p.kernelDims);
      final int[] inCoords = tensor.coordStream(true).mapToInt(kernelCoord -> {
        @Nonnull final int[] result = new int[o.getCoords().length];
        for (int index = 0; index < o.getCoords().length; index++) {
          final int outputCoordinate = o.getCoords()[index];
          final int kernelSize = p.kernelDims[index];
          final int baseCoordinate = Math.min(outputCoordinate * kernelSize, p.inputDims[index] - kernelSize);
          final int kernelCoordinate = kernelCoord.getCoords()[index];
          result[index] = baseCoordinate + kernelCoordinate;
        }
        return input.index(result);
      }).toArray();
      return new Tuple2<>(o.getIndex(), inCoords);
    }).collect(RefCollectors.toList());
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    final Result in = inObj[0];
    in.getData().length();

    @Nonnull final int[] inputDims = in.getData().getDimensions();
    final RefList<Tuple2<Integer, int[]>> regions = MaxPoolingLayer.calcRegionsCache
        .apply(new MaxPoolingLayer.CalcRegionsParameter(inputDims, kernelDims));
    final Tensor[] outputA = RefIntStream.range(0, in.getData().length())
        .mapToObj(dataIndex -> {
          final int[] newDims = RefIntStream.range(0, inputDims.length).map(i -> {
            return (int) Math.ceil(inputDims[i] * 1.0 / kernelDims[i]);
          }).toArray();
          return new Tensor(newDims);
        }).toArray(i -> new Tensor[i]);
    RefArrays.stream(outputA).mapToInt(x -> x.length()).sum();
    @Nonnull final int[][] gradientMapA = new int[in.getData().length()][];
    RefIntStream.range(0, in.getData().length()).forEach(dataIndex -> {
      @Nullable final Tensor input = in.getData().get(dataIndex);
      final Tensor output = outputA[dataIndex];
      @Nonnull final IntToDoubleFunction keyExtractor = inputCoords -> input.get(inputCoords);
      @Nonnull final int[] gradientMap = new int[input.length()];
      regions.parallelStream().forEach(tuple -> {
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
      });
      gradientMapA[dataIndex] = gradientMap;
    });
    return new Result(new TensorArray(outputA),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          if (in.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream
                .range(0, in.getData().length()).parallel().mapToObj(dataIndex -> {
                  @Nonnull final Tensor backSignal = new Tensor(inputDims);
                  final int[] ints = gradientMapA[dataIndex];
                  @Nullable final Tensor datum = data.get(dataIndex);
                  for (int i = 0; i < datum.length(); i++) {
                    backSignal.add(ints[i], datum.get(i));
                  }
                  return backSignal;
                }).toArray(i -> new Tensor[i]));
            in.accumulate(buffer, tensorArray);
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
  }

  public @Override
  @SuppressWarnings("unused")
  MaxPoolingLayer addRef() {
    return (MaxPoolingLayer) super.addRef();
  }

  public static @RefAware
  class CalcRegionsParameter {
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
