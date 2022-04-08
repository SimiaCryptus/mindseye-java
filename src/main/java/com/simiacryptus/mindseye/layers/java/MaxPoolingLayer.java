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
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.*;

/**
 * Class MaxPoolingLayer
 * A class that contains a static cache of a function that calculates regions.
 * It also contains an unused static Logger and an instance variable for kernel dimensions.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class MaxPoolingLayer extends LayerBase {

  private static final Function<MaxPoolingLayer.CalcRegionsParameter, RefList<Tuple2<Integer, int[]>>> calcRegionsCache = Util
      .cache(MaxPoolingLayer::calcRegions);
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxPoolingLayer.class);
  private int[] kernelDims;

  /**
   * Instantiates a new Max pooling layer.
   */
  protected MaxPoolingLayer() {
    super();
  }

  /**
   * Instantiates a new Max pooling layer.
   *
   * @param kernelDims the kernel dims
   */
  public MaxPoolingLayer(@Nonnull final int... kernelDims) {

    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  /**
   * Instantiates a new Max pooling layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected MaxPoolingLayer(@Nonnull final JsonObject id, @Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  /**
   * Creates a new {@link MaxPoolingLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   the map of character sequences to byte arrays
   * @return the new layer
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static MaxPoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  private static RefList<Tuple2<Integer, int[]>> calcRegions(final MaxPoolingLayer.CalcRegionsParameter p) {
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
    RefUtil.freeRef(inObj);
    final TensorList inData = in.getData();
    @Nonnull final int[] inputDims = inData.getDimensions();
    int length = inData.length();
    TensorArray data = fwd(inputDims, length);
    boolean alive = in.isAlive();
    int[][] gradientMap = getGradientMap(in.addRef(), inData, inputDims, length, data.addRef());
    Result.Accumulator accumulator = new Accumulator(inputDims, gradientMap, in.getAccumulator(), in.isAlive());
    in.freeRef();
    return new Result(data, accumulator, alive);
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

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
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

  private int[][] getGradientMap(Result in, TensorList inData, int[] inputDims, int length, TensorArray data) {
    final RefList<Tuple2<Integer, int[]>> regions = MaxPoolingLayer.calcRegionsCache
        .apply(new CalcRegionsParameter(inputDims, kernelDims));
    return RefIntStream.range(0, length).mapToObj(RefUtil.wrapInterface(dataIndex -> {
      @Nullable final Tensor input = inData.get(dataIndex);
      final Tensor output = ((TensorList) data).get(dataIndex);
      @Nonnull final IntToDoubleFunction keyExtractor = RefUtil.wrapInterface(input::get,
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
      }, output, input, keyExtractor));
      return gradientMap;
    }, in, data, regions, inData)).toArray(l -> new int[l][]);
  }

  @NotNull
  private TensorArray fwd(int[] inputDims, int length) {
    return new TensorArray(RefIntStream.range(0, length).mapToObj(dataIndex -> {
      return new Tensor(RefIntStream.range(0, inputDims.length).map(i -> {
        return (int) Math.ceil(inputDims[i] * 1.0 / kernelDims[i]);
      }).toArray());
    }).toArray(Tensor[]::new));
  }

  /**
   * This class represents the parameters for the calculation of regions.
   *
   * @param inputDims  the input dimensions
   * @param kernelDims the kernel dimensions
   * @docgenVersion 9
   */
  public static class CalcRegionsParameter {
    /**
     * The Input dims.
     */
    public final int[] inputDims;
    /**
     * The Kernel dims.
     */
    public final int[] kernelDims;

    /**
     * Instantiates a new Calc regions parameter.
     *
     * @param inputDims  the input dims
     * @param kernelDims the kernel dims
     */
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
      final MaxPoolingLayer.CalcRegionsParameter other = (MaxPoolingLayer.CalcRegionsParameter) obj;
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

  /**
   * The Accumulator class is used to hold input dimensions and gradient maps.
   * It also contains a Result.Accumulator object and a boolean value to
   * determine if it is alive.
   *
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private final int[][] gradientMapA;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputDims    the input dims
     * @param gradientMapA the gradient map a
     * @param accumulator  the accumulator
     * @param alive        the alive
     */
    public Accumulator(int[] inputDims, int[][] gradientMapA, Result.Accumulator accumulator, boolean alive) {
      this.inputDims = inputDims;
      this.gradientMapA = gradientMapA;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor backSignal = new Tensor(inputDims);
              final int[] ints = gradientMapA[dataIndex];
              @Nullable final Tensor datum = data.get(dataIndex);
              for (int i = 0; i < datum.length(); i++) {
                backSignal.add(ints[i], datum.get(i));
              }
              datum.freeRef();
              return backSignal;
            }, data)).toArray(Tensor[]::new));
        this.accumulator.accept(buffer, tensorArray);
      } else {
        if (null != buffer)
          buffer.freeRef();
        data.freeRef();
      }
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
